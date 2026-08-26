# 🧩 RE-ARC-3

RE-ARC-3 is an environment sampler for ARC-AGI-3.
It is designed as a simple install-and-play package that runs without API keys or external services.
It includes 100 packaged games in `v0.1.2`, including 4 official ARC-AGI-3 games plus local synthetic/community environments.
The repository also includes environment generation code (including Codex-assisted pipeline generation) for manual or automated creation of new environments.

## Batch Codex Runs

You can run the same Codex prompt across multiple game folders with `pipeline-codex-batch`.

Create a CSV jobs file such as:

```csv
game,folder
alternating_helpers,re_arc/environment_files/alternating_helpers/0001
snake,re_arc/environment_files/snake/0001
```

Then run:

```bash
pipeline-codex-batch \
  --jobs jobs.csv \
  --prompt-file prompt.txt \
  --max-workers 4 \
  --output-dir logs/codex-batch/helpers-reset-pass
```

Prompt templates support placeholders from each job row, including `{game}` and `{folder}`. Each run writes `prompt.txt`, `last_message.txt`, `stdout.log`, and `run.json` under the selected output directory, plus a top-level `summary.json`.

## Environment Authoring Contract

If you are adding or updating games:

- Read the [Environment Authoring Guide](docs/environment_authoring.md) first.
- For pipeline generation and Slurm usage, use the [Pipeline Guide](pipeline/README.md).
- After any code changes, run `make prepare` to run formatting, linting, type-checking, and tests.

## Use as a Package

Install:

Nightly (`master` branch):

```bash
python -m pip install "git+ssh://git@github.com/Tufalabs/re-arc-3.git"
```

Latest stable release (`v0.1.2`):

```bash
python -m pip install "git+ssh://git@github.com/Tufalabs/re-arc-3.git@v0.1.2"
```

List available games without creating a sampler:

```python
from re_arc import list_game_ids

games = list_game_ids()
print(f"Available games: {len(games)}")  # Available games: 100 (v0.1.2)
print("First 20:", games[:20])  # e.g. ('airtime_glider-0001', 'avoid_the_lava_timed_hazards-0001', ...)
```

Example:

```python
from re_arc import AugmentationConfig, EnvSampler

sampler = EnvSampler(
    include_tags=["official"],
    augment=True,
    augmentation_config=AugmentationConfig(
        color_permutation=False,
        rotation=True,
        flip_lr=True,
    ),
    seed=123,
)

env = next(sampler)  # same as sampler.sample(); uses rolling seed from `seed=123`
print("Env name:", env.name)

obs = env.reset()
done = False
while not done:
    action = env.action_space[0]
    obs, reward, done, info = env.step(action)
```

`augmentation_config` is optional and defaults to all augmentation components enabled.
You can also pass a dict in the same shape, e.g. `{"color_permutation": True, "rotation": False, "flip_lr": True}`.
`include_tags` and `exclude_tags` are optional and filter the discovered games by metadata tag before sampling.

### Metadata Tags

Each packaged game has a `tags` list in its `metadata.json`. These tags are mostly used by `include_tags`,
`exclude_tags`, and dataset curation:

- `keys-only`: the game is intended to be played with keyboard/action inputs only.
- `click-only`: the game is intended to be played with click inputs only.
- `click-keys`: the game uses both click inputs and keyboard/action inputs.
- `official`: an official ARC-AGI-3 game environment.
- `official-reimplemented`: a close local reconstruction of an official ARC-AGI-3 game.
- `arc-like`: a synthetic ARC-like game.
- `easy`, `medium`, `hard`: rough difficulty labels. These are separate from origin/style tags.
- `debugging`: generated/debug-oriented environments used for robustness and tooling coverage.
- `no-time-bar`: the game does not display a visible time/step meter.

Tags are flat labels, not inherited classes. For example, an easy synthetic ARC-like game should use
both `arc-like` and `easy`; an easy official reconstruction should use both `official-reimplemented`
and `easy`. Deprecated compound tags such as `arc-like-easy`, `arc-like-medium`, `official-easy`,
`official-medium`, and `official-like` should not be used for new metadata.

You can also save named datasets as JSON files instead of tagging each game.
Dataset files may be a list of game ids, or an object with `name`, optional `tags`, and `game_ids`:

```json
{
  "name": "train",
  "tags": ["optimisation"],
  "game_ids": ["maze-0001", "push-0001", "easy_taps-0001"]
}
```

Then load them by file name, dataset name, dataset tag, or path:

```python
train = EnvSampler(datasets="train", seed=123)
eval_games = list_game_ids(datasets="eval")
```

For CLI sampling, set `GAME_ID=random` and select the dataset:

```bash
re-arc --game random --dataset train
```

The package includes `train` and `eval` datasets. They are a one-time random 70/30 split of the current
packaged catalog, with 8 official games in `eval` and all remaining official games in `train`. To load a
tagged subset of either split, combine the dataset filter with metadata tags:

```python
eval_official = list_game_ids(datasets="eval", include_tags="official")
```

For `config.env` or CLI-driven play, `AUGMENT=true` enables sampler augmentation. You can selectively disable
components with:

```text
AUGMENT=true
AUGMENT_COLOR_PERMUTATION=false
AUGMENT_ROTATION=true
AUGMENT_FLIP_LR=true
```

`make play` and `re-arc --policy web` honor these flags for `GAME_ID=random` and for local packaged games.

`exclude_datasets` can remove a held-out set from a broader training set. Dataset filters are applied before
`include_tags` and `exclude_tags`, so metadata tags can still narrow a saved dataset.

Reward contract is always transition-based:
- each level transition contributes `delta_levels / win_levels`
- terminal `WIN` reconciles the total episode reward to `1.0`
- samplers and CLI already return reward-aware envs; do not add an extra reward wrapper on top

`EnvSampler` loads packaged environments by default. If needed, pass `environments_dir="..."` to use a custom environment folder.

## Replay Viewer

`re-arc` can write replay traces alongside DSL-generated GIFs and serve them in a local browser UI.

```bash
re-arc --game identify_the_agent-0001 --policy dsl --gif out/identify.gif
# writes out/identify.gif and out/identify.replay.json
```

```bash
re-arc --policy replay --replay-dir docs/replays --port 8000
# open http://127.0.0.1:8000/
```

For the full committed gallery, see [docs/replays.md](docs/replays.md).

## Demo GIFs

The following examples were generated from this repo and are committed under `docs/replays/`:

<table>
  <tr>
    <td align="center"><strong>Flux</strong><br><img src="docs/replays/flux-dsl.gif" width="240" /></td>
    <td align="center"><strong>Frogger</strong><br><img src="docs/replays/frogger-dsl.gif" width="240" /></td>
    <td align="center"><strong>Keys</strong><br><img src="docs/replays/keys-dsl.gif" width="240" /></td>
  </tr>
  <tr>
    <td align="center"><strong>Glyph</strong><br><img src="docs/replays/glyph-dsl.gif" width="240" /></td>
    <td align="center"><strong>Maze</strong><br><img src="docs/replays/maze-dsl.gif" width="240" /></td>
    <td align="center"><strong>Push</strong><br><img src="docs/replays/push-dsl.gif" width="240" /></td>
  </tr>
  <tr>
    <td align="center"><strong>SWCH</strong><br><img src="docs/replays/swch-dsl.gif" width="240" /></td>
    <td align="center"><strong>Taps</strong><br><img src="docs/replays/taps-dsl.gif" width="240" /></td>
    <td align="center"><strong>Turn</strong><br><img src="docs/replays/turn-dsl.gif" width="240" /></td>
  </tr>
</table>
