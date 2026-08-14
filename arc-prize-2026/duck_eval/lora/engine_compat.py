"""Two loader fixes needed to run Tufa's own re-arc-3 environment pool locally.

`arc_agi.local_wrapper.LocalEnvironmentWrapper._load_game_class` works for the public
25 but fails on most of the 165 non-public families for two reasons that have
nothing to do with the games themselves:

1. It only looks for `<ClassName>.py` / `<classname>.py`. Many re-arc-3
   families ship `<family_name>.py` (`axis_reflect.py` for `AxisReflect`).
2. It `exec`s the source into a module that is never registered in
   `sys.modules`. Any game source that uses `@dataclass` then dies inside
   `dataclasses._is_type` with
   `AttributeError: 'NoneType' object has no attribute '__dict__'`,
   because dataclasses resolves `cls.__module__` through `sys.modules`.

`apply()` is idempotent and only touches the loader. Game logic, action
semantics, scoring and baselines are untouched, so a plan verified under this
patch is a plan the unpatched engine would also accept -- and the public 25
load identically with or without it (asserted in `selftest.py`).
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

_APPLIED = False


def apply() -> bool:
    global _APPLIED
    if _APPLIED:
        return False
    from arc_agi.local_wrapper import LocalEnvironmentWrapper

    def _load_game_class(self, seed: int = 0) -> None:
        info = self.environment_info
        if info.local_dir is None:
            return
        local_dir = Path(info.local_dir)
        class_name = info.class_name
        if not class_name:
            self.logger.error(f"class_name not set for {info.game_id}")
            return

        family = str(info.game_id).rsplit("-", 1)[0]
        candidates = [
            local_dir / f"{class_name.lower()}.py",
            local_dir / f"{class_name}.py",
            local_dir / f"{family}.py",                      # FIX 1
            local_dir / f"{family.replace('_', '')}.py",
        ]
        game_file = next((p for p in candidates if p.exists()), None)
        if game_file is None:
            extra = sorted(p for p in local_dir.glob("*.py") if p.name != "__init__.py")
            game_file = extra[0] if len(extra) == 1 else None
        if game_file is None:
            self.logger.error(
                f"Game source file not found for {info.game_id} in {local_dir}"
            )
            return

        module_name = f"arc_agi_3.{info.game_id}".replace("-", "_")
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module                    # FIX 2
        try:
            exec(game_file.read_text(encoding="utf-8"), module.__dict__)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            self.logger.error(f"Error executing game source for {info.game_id}: {exc}")
            return

        cls = getattr(module, class_name, None)
        if cls is None or not isinstance(cls, type):
            self.logger.error(f"class {class_name} not found in {game_file}")
            return
        sig = inspect.signature(cls)
        kwargs = {"seed": seed} if (seed is not None and "seed" in sig.parameters) else {}
        self._game = cls(**kwargs)

    LocalEnvironmentWrapper._load_game_class = _load_game_class
    _APPLIED = True
    return True
