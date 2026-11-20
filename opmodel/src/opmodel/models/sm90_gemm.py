from dataclasses import dataclass, replace
from functools import lru_cache
import json
import hashlib
from pathlib import Path
import re

from opmodel.api import DType
from opmodel import settings as _settings

EVIDENCE = Path(__file__).resolve().parents[1] / "configs/gemm_catalogs/sm90_evidence"
FORCED = EVIDENCE / "lt_heur_forced_h100.csv"
CUTLASS = "CUTLASS v3.5.1 f7b19de32c5d1f3cedfc735c2849f12b537522ee"
TILES_SOURCE = ("distributed_runs/h100_counts/launches/05a0baefeecb40ffa5a00444f687e1a4/"
                "results/session/stages/cublaslt-query/output/tiles.json")
TILES_SHA256 = "0c1e3c4c9b59b0885fd411e6003631734599bfe4ff27298f25805cec7c18088b"

@dataclass(frozen=True)
class WgmmaCounts:
    issued_flops: int
    operand_read_bytes: int
    epilogue_write_bytes: int
    instruction_count: int
    swapped: bool

def tile_instruction_counts(m, n, k, *, swapped, input_bytes=2, output_bytes=2, nominal_n=None):
    if min(m, n, k) <= 0:
        return WgmmaCounts(0, 0, 0, 0, swapped)
    logical_m, logical_n = (n, m) if swapped else (m, n)
    m_atoms = (logical_m + 63) // 64
    if _settings.get("OPMODEL_SM90_CUTLASS_N_SELECTOR"):
        selector_n = nominal_n if nominal_n is not None else ((logical_n + 7) // 8) * 8
        atom_n = next(x for x in (256, 192, 128, 96, 64, 32, 16, 8) if selector_n % x == 0)
        n_atoms = (logical_n + atom_n - 1) // atom_n
        n_padded = n_atoms * atom_n
    else:
        n_padded = ((logical_n + 7) // 8) * 8
        n_atoms = (n_padded + 255) // 256
    k_atoms = (k + 15) // 16
    instructions = m_atoms * n_atoms * k_atoms
    issued = 2 * m_atoms * 64 * n_padded * k_atoms * 16
    reads = m_atoms * k_atoms * (n_atoms * 64 + n_padded) * 16 * input_bytes
    return WgmmaCounts(int(issued), int(reads), int(m*n*output_bytes),
                      instructions, swapped)

def instruction_counts(problem, kernel):
    from opmodel.ops import dtype_nbytes
    from opmodel.split_k import normalize_split_k, split_k_lengths
    swapped = problem.m < 64 and problem.m <= problem.n
    def groups(size, tile):
        full, tail = divmod(size, tile)
        return ((tile, full), (tail, int(tail > 0)))
    lengths = (problem.k,)
    if kernel.split_k > 1:
        lengths = split_k_lengths(problem.k, kernel.split_k, kernel.cta_k,
                                  kernel.reduction_scheme)

    output_bytes = dtype_nbytes(problem.output_dtype)
    if kernel.split_k > 1 and normalize_split_k(kernel.split_k, kernel.reduction_scheme)[1] == 'compute_type':
        output_bytes = 4
    issued = reads = instructions = 0
    for length in lengths:
        for m, mc in groups(problem.m, kernel.cta_m):
            for n, nc in groups(problem.n, kernel.cta_n):
                if not mc or not nc or not length:
                    continue
                counts = tile_instruction_counts(m, n, length, swapped=swapped,
                    input_bytes=dtype_nbytes(problem.input_dtype), output_bytes=output_bytes,
                    nominal_n=kernel.cta_m if swapped else kernel.cta_n)
                count = problem.batch * mc * nc
                issued += count * counts.issued_flops
                reads += count * counts.operand_read_bytes
                instructions += count * counts.instruction_count
    ctas = problem.batch * ((problem.m + kernel.cta_m - 1) // kernel.cta_m) * (
        (problem.n + kernel.cta_n - 1) // kernel.cta_n)

    epilogue = ctas * sum(length > 0 for length in lengths) * kernel.cta_m * kernel.cta_n * output_bytes
    return WgmmaCounts(issued, reads, int(epilogue), instructions, swapped)

def stage_count(tile, hardware):
    capacity = hardware.compute.shared_memory_bytes_per_block
    if capacity is None or capacity <= 0:
        raise ValueError("sm90 StageCountAutoCarveout requires per-block SMEM capacity")
    m, n, k = tile
    stages = (capacity - 1) // (2 * k * (m + n) + 16)
    if stages < 1:
        raise ValueError(f"sm90 tile {tile} cannot fit one pipeline stage")
    return stages

def _template(name, tile, hardware, source, split_k=1, reduction_scheme="none"):
    from .extended_roofline import GemmKernelTemplate
    m, n, k = tile

    return GemmKernelTemplate(
        name=name, cta_m=m, cta_n=n, cta_k=k,
        warp_m=64, warp_n=8, warp_k=k, num_warp_tile_k=1,
        mma_m=64, mma_n=8, mma_k=16,
        pipeline_stages=stage_count(tile, hardware),
        warps_per_cta=4, registers_per_thread=0,
        source=f"{source}; {CUTLASS}; docs/h100_b5_sources.md; "
               "SS WGMMA m64nNk16; docs/h100_wgmma_fix.md",
        dtypes=(DType.BF16,), split_k=split_k, reduction_scheme=reduction_scheme,
        architecture="sm90")

@lru_cache(maxsize=1)
def _package():
    raw = (EVIDENCE / "tiles.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TILES_SHA256:
        raise ValueError("H100 tiles.json source SHA256 mismatch")
    caps = json.loads(raw)
    index, algo = next((i, a) for i, a in enumerate(caps["algorithms"]) if a["algo_id"] == 66)
    sources = {}
    for i, tile in enumerate(algo["tiles"]):
        m, n = map(int, re.fullmatch(r"CUBLASLT_MATMUL_TILE_(\d+)x(\d+)", tile["name"]).groups())
        for logical in {(m, n, 64), (n, m, 64)}:
            sources.setdefault(logical, []).append(f"/algorithms/{index}/tiles/{i}")
    manifest = json.loads((EVIDENCE / "candidate_manifest.json").read_text())
    return sources, manifest

def catalog(dtype, hardware, name):
    if dtype != DType.BF16:
        raise ValueError("sm90 cuBLAS package supports BF16 only")
    if name not in ("", "default", "cublaslt_sm80_bf16", "cublaslt_sm80_bf16_splitk"):
        raise ValueError(f"no sm90 GEMM description for catalog {name!r}")
    sources, manifest = _package()
    templates = tuple(_template(f"lt_sm90_bf16_{m}x{n}x{k}_AUTO", (m, n, k), hardware,
        f"{TILES_SOURCE}#{','.join(sources[(m, n, k)])}; sha256={TILES_SHA256}; "
        "local copy sm90_evidence/tiles.json; algo 66 caps; both logical M/N orientations; registers unused")
        for m, n, k in sorted(sources))
    if name != "cublaslt_sm80_bf16_splitk":
        return templates
    by_name = {t.name: t for t in templates}

    return templates + tuple(replace(by_name[c["base_name"]], name=c["name"],
        split_k=c["split_k"], reduction_scheme=c["reduction_scheme"],
        source=by_name[c["base_name"]].source + "; observed-only for algo 66; caps flags not measured")
        for c in manifest["candidates"])

def resolve_forced(shape, role, hardware):
    from .extended_roofline import _load_forced_templates

    rows = _load_forced_templates(str(FORCED)).get(tuple(shape), [])
    rows = [r for r in rows if r["gemm_role"] == role]
    configs = {(r["model_cta_m"], r["model_cta_n"], r["model_cta_k"],
                r["splitK"], r["reduction_scheme"]) for r in rows}
    if len(configs) > 1:
        raise ValueError(f"ambiguous H100 forced key {tuple(shape)}, {role}: "
                         + ",".join(r["shape_id"] for r in rows))
    if not rows:
        return None, []
    row = rows[0]
    tile = tuple(int(row[k]) for k in ("model_cta_m", "model_cta_n", "model_cta_k"))
    return _template("lt_heur_" + row["shape_id"], tile, hardware,
                     "lt_heur_forced_h100.csv: " + row["heuristic_source"],
                     int(row["splitK"]), int(row["reduction_scheme"])), rows
