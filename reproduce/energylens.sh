#!/bin/sh
set -u
. "$(dirname -- "$0")/env.sh"
O=$OUT/energylens
mkdir -p "$O/logs"
rc=0
start=$(date +%s)
"$PY" -B -m paper_pipeline.driver run energylens --workers "$WORKERS" > "$O/logs/run.log" 2>&1 || rc=1
"$PY" -B -m paper_pipeline.driver score energylens > "$O/logs/score.log" 2>&1 || rc=1
echo "[energylens] run + score ($(( $(date +%s) - start )) s): $(tail -n 1 "$O/logs/run.log")"
"$PY" -B reproduce/scripts/energylens_vsraw.py "$O/energylens_vsraw_cells.csv" > "$O/logs/vsraw.log" 2>&1 || rc=1
"$PY" -B reproduce/scripts/energylens_figure.py "$O" > "$O/logs/figure.log" 2>&1 || rc=1
sed -n '/MAPE %/,$p' "$O/logs/figure.log"
echo "[energylens] figure: reproduce/out/energylens/energylens.pdf; rc=$rc"
exit $rc
