#!/usr/bin/env bash
# A/B: campaign serve with CUDA graphs vs --enforce-eager.
# Usage: ab_graph_vs_eager.sh MODEL_DIR [SERVED_NAME]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# repo root when script lives in scripts/
DIR="${1:?model dir}"
NAME="${2:-tc-ab-graph}"
PORT="${PORT:-8003}"
OUT="${OUT_DIR:-$ROOT/results}"
mkdir -p "$OUT"
LOG="$OUT/ab_graph_eager.log"
L(){ echo "[AB $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

kill_port() {
  local pids
  pids=$(ss -lntp 2>/dev/null | awk -v p=":$PORT" '$0 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
  for pid in $pids; do
    local pgid; pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && [ "$pgid" != "0" ] && kill -9 -"$pgid" 2>/dev/null || true
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 2
  pkill -9 -f 'VLLM::Worker' 2>/dev/null || true
  sleep 2
}

wait_health() {
  local slog=$1 i
  for i in $(seq 1 90); do
    if curl -sf -m5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then return 0; fi
    if [ -f "$slog" ] && grep -qiE 'RuntimeError|Engine core initialization failed|CUDA out of memory|ValueError:|Traceback \(most recent' "$slog"; then
      L "FATAL"; tail -40 "$slog" | tee -a "$LOG"; return 1
    fi
    sleep 10
  done
  return 1
}

run_arm() {
  local mode=$1  # graph | eager
  local slog="$OUT/serve_${mode}.log"
  local flag=()
  [ "$mode" = "eager" ] && flag=(--eager)
  L "=== boot mode=$mode ==="
  kill_port
  nohup bash "$ROOT/scripts/sm70_nvfp4_campaign_serve.sh" "$DIR" "$NAME" "${flag[@]}" >"$slog" 2>&1 &
  if ! wait_health "$slog"; then
    L "BOOT_FAIL $mode"
    echo "BOOT_FAIL" > "$OUT/${mode}.result"
    return 1
  fi
  L "healthy — smoke $mode"
  if python3 "$ROOT/scripts/sm70_nvfp4_smoke_paris.py" --url "http://127.0.0.1:${PORT}" --model "$NAME" --label "$mode" 2>&1 | tee -a "$LOG" | tee "$OUT/${mode}.smoke"; then
    echo "PASS" > "$OUT/${mode}.result"
    L "PASS $mode"
  else
    echo "FAIL" > "$OUT/${mode}.result"
    L "FAIL $mode"
  fi
  kill_port
}

: > "$LOG"
L "model=$DIR served=$NAME"
# verify overlay if possible
python3 "$ROOT/scripts/sm70_nvfp4_verify_identity.py" 2>&1 | tee -a "$LOG" || L "verify soft-fail (continue)"

run_arm graph || true
run_arm eager || true

python3 - <<PY | tee -a "$LOG"
import pathlib
out = pathlib.Path("$OUT")
print("\n# Graph vs eager summary")
print("| mode | result | notes |")
print("|------|--------|-------|")
for mode in ("graph", "eager"):
    r = (out / f"{mode}.result").read_text().strip() if (out / f"{mode}.result").exists() else "missing"
    smoke = (out / f"{mode}.smoke").read_text().replace("\n", " ")[:120] if (out / f"{mode}.smoke").exists() else ""
    print(f"| {mode} | {r} | {smoke} |")
print("""
Interpretation:
- graph PASS + eager PASS → match pve3; JJ should use campaign flags (not wheel)
- graph FAIL + eager PASS → CUDA-graph path bug / config; use --enforce-eager workaround + file 1Cat issue
- both FAIL → overlay/checkpoint/env (check normalize hashes, VLLM_SM70_NVFP4_TURBOMIND=1, TINY scales)
- graph PASS + eager FAIL → unexpected; collect logs
""")
PY
L "=== AB COMPLETE ==="
