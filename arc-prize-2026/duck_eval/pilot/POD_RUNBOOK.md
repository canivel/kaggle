# Phase-0c Synthesis Pilot -- RunPod A40 Runbook (ONE pod session)

One-time budget: everything below runs in a single pod session, then the pod
is TERMINATED. Steps are ordered; do not skip the mock smoke (step 5) before
the real run.

Estimated wall-clock: 6-12 h (10 games x 3 scaffolds x 2 regimes x <=4 LLM
calls = <=240 calls; at A40 27B speeds ~1.5-3 min/call). Cost at ~$0.40/h
spot: ~$3-6. Terminate as soon as results are pulled.

## 0. Local prep (Windows, before paying for anything)

```powershell
cd f:\kaggle\arc-prize-2026
uv run python duck_eval/pilot/run_pilot.py --mock --games bp35,r11l --scaffolds freeform   # must PASS
uv run python duck_eval/pilot/package_data.py                                              # writes duck_eval/pilot_bundle.tar.gz
```

## 1. Create the pod (A40, secure cloud, spot OK)

```bash
runpodctl create pod \
  --name arc-pilot-0c \
  --gpuType "NVIDIA A40" \
  --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 40 \
  --volumeSize 120 \
  --volumePath /workspace \
  --secureCloud \
  --bid 0.44 \
  --ports "22/tcp"
runpodctl get pod            # note POD_ID; wait for RUNNING
```

Get SSH coordinates (host/port) from `runpodctl get pod <POD_ID> -a` or the
RunPod console SSH tab. Below: `$SSH = user@host -p PORT` per console.

## 2. Transfer the bundle (one scp)

```powershell
scp -P <PORT> f:\kaggle\arc-prize-2026\duck_eval\pilot_bundle.tar.gz root@<HOST>:/workspace/
scp -P <PORT> f:\kaggle\arc-prize-2026\duck_eval\provision_a40.sh      root@<HOST>:/workspace/
```

## 3. Provision (on the pod) -- vLLM + Qwen3-27B

The pilot only needs the vLLM part of `provision_a40.sh` (model download +
server + parity probe). The taaf_bundle editable installs are NOT needed --
skip them by running the excerpt:

```bash
ssh -p <PORT> root@<HOST>
cd /workspace
tar xzf pilot_bundle.tar.gz
pip install -q uv && uv pip install --system "vllm>=0.8" huggingface_hub httpx numpy

# model download + vLLM server + tokens/s probe: reuse provision_a40.sh's
# model/vLLM/probe sections verbatim (they are self-contained after deps):
sed -n '/--- model/,$p' provision_a40.sh > provision_vllm.sh
bash provision_vllm.sh          # downloads Qwen3-27B, starts vLLM on :8000, prints TOKENS_PER_SECOND

curl -sf http://127.0.0.1:8000/v1/models   # sanity: model listed
```

Record the printed `TOKENS_PER_SECOND=` line (feeds the Phase-0b parity
measurement).

## 4. Pod-side mock smoke (CPU, ~1 min -- proves the pipeline before burning GPU time)

```bash
cd /workspace/pilot_bundle
python pilot/run_pilot.py --mock --games bp35,r11l --scaffolds freeform
# expect: summary JSON printed, results_mock/*.json written, no traceback
rm -rf pilot/results_mock
```

## 5. Real run (background, resumable)

```bash
cd /workspace/pilot_bundle
nohup python pilot/run_pilot.py \
  --endpoint http://127.0.0.1:8000/v1 \
  --out-dir /workspace/pilot_results \
  > /workspace/pilot.log 2>&1 &
echo $! > /workspace/pilot.pid
```

Full matrix: 10 games x 3 scaffolds x 2 regimes (capped6k deciding +
uncapped32k upper-bound), <=4 refactor iterations each. Results are written
per (game, scaffold) immediately; if the pod is preempted and restarted, the
same command SKIPS finished pairs and resumes.

## 6. Monitor

```bash
tail -f /workspace/pilot.log                          # per-arm progress lines
ls /workspace/pilot_results | wc -l                   # 30 files + summary.json when done
watch -n 60 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
grep -c '"class_a": true' /workspace/pilot_results/*.json
```

If vLLM dies (OOM etc.): `tail /workspace/vllm.log`, restart via the vLLM
block of provision_vllm.sh, then re-run step 5 (it resumes).

## 7. Pull results back (from the LOCAL machine)

```powershell
scp -P <PORT> -r root@<HOST>:/workspace/pilot_results f:\kaggle\arc-prize-2026\duck_eval\pilot\
scp -P <PORT> root@<HOST>:/workspace/pilot.log        f:\kaggle\arc-prize-2026\duck_eval\pilot\pilot_results\
scp -P <PORT> root@<HOST>:/workspace/vllm.log         f:\kaggle\arc-prize-2026\duck_eval\pilot\pilot_results\
```

Verify locally BEFORE terminating:

```powershell
uv run python duck_eval/pilot/run_pilot.py --summary-only --out-dir duck_eval/pilot/pilot_results
```

Confirm `summary.json` reports `n_games_class_a_under_logo_capped6k` and all
30 (game, scaffold) files are present.

## 8. TERMINATE THE POD (one-time budget -- do not leave it running)

```bash
runpodctl remove pod <POD_ID>
runpodctl get pod              # confirm the pod is GONE
```

Then check the RunPod console billing page shows the pod stopped accruing.

## Decision readout (against winning_solution_FINAL.md)

- Phase-2 entry gate: >=4/10 games Class-A on the capped6k LOGO numbers
  (`summary.json: n_games_class_a_under_logo_capped6k`).
- P0 kill: <4/10 -> Phase 2 pre-killed.
- Also record: generation-length p90 + truncation rate per scaffold (pins the
  Phase-2 headroom line), tokens_to_first_class_a per game, memorization
  gaps, and the uncapped32k matrix (upper bound only; decides nothing).
- Pre-registered caveat (in every result JSON): observations are level-0
  random-exploration histories, not duck trajectories.
- NOT in this kit (separate steps per the FINAL doc): the >=2-game
  closed-loop arm and the RTX-PRO-6000 quantization-anchor battery.
