#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
if ! git rev-parse --verify -q HEAD >/dev/null 2>&1; then
  git init -q .
  git add -A
  git -c user.name=artifact -c user.email=artifact@example.invalid commit -q -m "artifact"
fi
for a in data/archives/*.tar.xz; do
  echo "extracting $a"
  tar -xJf "$a" -C data
done
if [ -n "${PROTOBUF_ROOT:-}" ]; then
  export PATH="$PROTOBUF_ROOT/bin:$PATH" Protobuf_ROOT="$PROTOBUF_ROOT"
  export CMAKE_PREFIX_PATH="$PROTOBUF_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export LD_LIBRARY_PATH="$PROTOBUF_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
command -v protoc >/dev/null 2>&1 || { echo "protoc not found: install protobuf or set PROTOBUF_ROOT" >&2; exit 1; }
echo "building AstraSim (analytical, congestion aware) with $(protoc --version), $(${CXX:-c++} --version | head -n 1)"
bash astra-sim/build/astra_analytical/build.sh -t congestion_aware
B=astra-sim/build/astra_analytical/build/bin
cp "$B/AstraSim_Analytical_Congestion_Aware" "$B/AstraSim_Analytical_Congestion_Aware.7a1fbae5"
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON; elif [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
"$PY" -c "import numpy, yaml, matplotlib, google.protobuf, graphviz, scotchpy" || {
  echo "missing Python packages: pip install -r requirements.txt" >&2; exit 1; }
"$PY" -c "import sys; sys.path.insert(0, 'astra-sim/extern/graph_frontend/chakra/schema/protobuf'); import et_def_pb2" || {
  echo "et_def_pb2.py does not load with this protobuf package" >&2; exit 1; }
echo "setup done"
