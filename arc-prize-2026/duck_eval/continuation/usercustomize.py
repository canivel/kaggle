"""Interpreter-startup hook that applies the game-over-continuation patch.

Python imports ``usercustomize`` automatically at startup when its directory is
on ``PYTHONPATH`` and site processing is enabled. The gpt56 probe rig runs the
duck harness in a child process (``python -m inference.framework.run``), so the
patch has to be applied *inside that child* before any session builds its system
prompt. run_probe.py prepends this directory to the child's PYTHONPATH so this
hook fires; the CONT_FIX gate is translated to the CONTINUATION_DISABLE kill
switch honored by continuation_patch.apply().

Any failure here is swallowed -> stock harness (vanilla fallback).
"""
try:
    import continuation_patch

    continuation_patch.apply()
except Exception:  # noqa: BLE001 - never break interpreter startup
    pass
