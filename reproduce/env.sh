ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x "$ROOT/.venv/bin/python" ]; then PY=$ROOT/.venv/bin/python
else PY=python3
fi
NCPU=$(nproc 2>/dev/null || echo 8)
WORKERS=${WORKERS:-$(( NCPU < 32 ? NCPU : 32 ))}
OUT=$ROOT/reproduce/out
export CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=$OUT/.mpl
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT" || exit 1
if [ ! -x astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware.7a1fbae5 ] || \
   [ ! -d data/workload_set/counts_fullstep/cap ]; then
  echo "run reproduce/setup.sh first" >&2
  exit 1
fi
