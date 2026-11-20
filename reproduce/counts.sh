#!/bin/sh
set -u
. "$(dirname -- "$0")/env.sh"
O=$OUT/counts
if [ -e "$O" ]; then echo "reproduce/out/counts exists; move it away to rerun" >&2; exit 2; fi
mkdir -p "$O/logs"
S=reproduce/scripts
A100=data/workload_set/counts_heldout
H100=data/workload_set/counts_h100_heldout
rc=0
step() {
  name=$1; ok=$2; shift 2
  start=$(date +%s)
  "$PY" -B "$@" > "$O/logs/$name.log" 2>&1
  r=$?
  echo "[counts] $name rc=$r ($(( $(date +%s) - start )) s)"
  case " $ok " in *" $r "*) ;; *) rc=1 ;; esac
}
step a100_new "0 1" $S/counts_predict_a100.py "$O/pred/pred_new" --only new --plan $A100/PLAN_v1.csv --workers "$WORKERS"
step a100_supp1 0 $S/counts_predict_a100.py "$O/pred/pred_supp1" --only new --plan $A100/PLAN_supp1.csv --workers "$WORKERS"
step a100_old 0 $S/counts_predict_a100.py "$O/pred/pred_old" --only old --workers "$WORKERS"
step a100_moe 0 $S/counts_predict_moe.py "$O/pred/pred_moe" --workers "$WORKERS"
step h100_new 0 $S/counts_predict_h100.py "$O/pred/pred_h100" --plan $H100/PLAN_H100.csv --workers "$WORKERS"
step h100_add 0 $S/counts_predict_h100.py "$O/pred/pred_h100_add" --plan $H100/PLAN_H100_ADD.csv --workers "$WORKERS"
step a100_score 0 $S/counts_score_a100.py "$O"
step h100_score 0 $S/counts_score_h100.py "$O" --workers "$WORKERS"
"$PY" -B $S/counts_figure.py "$O" > "$O/logs/figure.log" 2>&1 || rc=1
sed -n '/MAPE %/,$p' "$O/logs/figure.log"
echo "[counts] figure: reproduce/out/counts/counts_A100_H100.pdf; rc=$rc"
exit $rc
