"""Train / holdout split over environment FAMILIES.

Binding rule for this lane (`learnings/war_room/lora_lane_2026-08-13.md` R1):

    A family that is scored on the Kaggle leaderboard is NEVER a training
    source. The 25 public families are eval-only, forever.

That leaves the 165 non-public re-arc-3 families in Tufa's own bundle as the
entire training pool, which is then split again:

    TRAIN   -- 80% of the non-public families; the adapter sees these.
    DEV     -- 20% of the non-public families; never trained on. Used to read
               "did the policy transfer to an unseen family?" for free, on CPU,
               before any GPU or Kaggle spend.
    EVAL    -- the 25 public families. Never trained on, never dev-tuned on.
               Only ever measured against, and only through the real harness.

Split is by family name hash, so it is stable across runs and machines and does
not drift when new families appear.
"""
from __future__ import annotations

import hashlib

from harness_env import INTERNAL_ENVS, PUBLIC_ENVS, family, list_environments

DEV_FRACTION = 0.20


def _bucket(name: str) -> float:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def public_families() -> set[str]:
    return {family(g) for g in list_environments(PUBLIC_ENVS)}


def build_split() -> dict[str, list[str]]:
    """Returns {'train': [game_id...], 'dev': [...], 'eval': [...]}"""
    public = public_families()
    internal = list_environments(INTERNAL_ENVS)

    train: list[str] = []
    dev: list[str] = []
    for game_id in internal:
        fam = family(game_id)
        if fam in public:
            # Same family as a scored game -> excluded from training entirely,
            # even though this is a different generated instance.
            continue
        (dev if _bucket(fam) < DEV_FRACTION else train).append(game_id)

    return {
        "train": sorted(train),
        "dev": sorted(dev),
        "eval": sorted(list_environments(PUBLIC_ENVS)),
    }


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    split = build_split()
    print(json.dumps({k: len(v) for k, v in split.items()}, indent=2))
    print("train sample:", split["train"][:8])
    print("dev  sample:", split["dev"][:8])
