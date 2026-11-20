#!/bin/sh
set -u
. "$(dirname -- "$0")/env.sh"
O=$OUT/energy
mkdir -p "$O/logs"
rc=0
for e in bare_step96 multi_gpu32_locked1065 megatron_single_a100 megatron_multi_a100; do
  start=$(date +%s)
  "$PY" -B -m paper_pipeline.driver run "$e" --workers "$WORKERS" > "$O/logs/$e.run.log" 2>&1 || rc=1
  "$PY" -B -m paper_pipeline.driver score "$e" > "$O/logs/$e.score.log" 2>&1 || rc=1
  echo "[energy] $e ($(( $(date +%s) - start )) s): $(tail -n 1 "$O/logs/$e.run.log")"
done
"$PY" -B reproduce/scripts/energy_figures.py "$O" > "$O/logs/figure.log" 2>&1 || rc=1
sed -n '/MAPE %/,$p' "$O/logs/figure.log"
echo "[energy] figures: reproduce/out/energy/single_gpu_energy.pdf, multi_gpu_energy.pdf; rc=$rc"
exit $rc
