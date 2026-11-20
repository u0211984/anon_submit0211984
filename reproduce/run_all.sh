#!/bin/sh
set -u
D=$(dirname -- "$0")
rc=0
for s in counts energy energylens; do
  sh "$D/$s.sh" || rc=1
done
if [ $rc -eq 0 ]; then echo "all validation numbers match the paper"; else echo "some checks failed (see reproduce/out/*/logs)"; fi
exit $rc
