"""
verdicts.py -- turn a checker VERDICT into an INSTRUCTION, and name the failure class.

The lesson this repo keeps re-learning: "X is wrong" teaches the model nothing, "change exactly Y"
does. upstream agent.py did this in `enrich()`, a hard-coded chain of regexes. This module is the
same knowledge as an explicit table, so that

  * every failure lands in a named BUCKET -- the failure taxonomy (scripts/taxonomy.py) and the
    error-keyed doc retrieval (doc_slices.json) are both keyed by it;
  * each rule is unit-tested (python verdicts.py --selftest);
  * adding a translation is one row, not another elif.

    bucket, text = translate(raw_error)   # raw simulator/loader error -> (bucket, error + instruction)
    bucket = classify(feedback)           # any feedback string agent.grade() produced -> bucket

The instructions are upstream's wording unless a comment says otherwise; moving them here does not
change what the model is told.
"""

import re


def _available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'. Needs the SDK; empty without it."""
    import difflib
    import importlib
    mod_name, _, attr = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return ""
    names = [n for n in dir(mod) if not n.startswith("_")]
    close = difflib.get_close_matches(attr, names, n=6, cutoff=0.4)
    if close:
        return (f" `{mod_name}` has no `{attr}`. The closest real names are: "
                f"{', '.join(close)}. Pick one of those or use a different approach.")
    return (f" `{mod_name}` has no `{attr}`, and nothing similar exists. Its real names include: "
            f"{', '.join(sorted(names)[:25])}.")


def _real_signature(func_name):
    import inspect
    for mod_name in ("nki.language", "nki.isa", "nki"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        fn = getattr(mod, func_name, None)
        if fn is None:
            continue
        try:
            return f"{mod_name.split('.')[-1]}.{func_name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            return f"{mod_name.split('.')[-1]}.{func_name}"
    return ""


# ---------------------------------------------------------------- raw error -> instruction
#
# (bucket, pattern, instruction builder). First match wins, so order matters exactly as it did in
# enrich(): the specific dma/psum patterns sit above the generic "must be in" one.

def _i_memregion(m, t):
    return (" nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not functions. Do not call "
            "them. Allocate with nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the "
            "region as the buffer= argument.")


def _i_kwarg(m, t):
    sig = _real_signature(m.group(1))
    return (f" Remove the `{m.group(2)}=` argument."
            + (f" The real signature is {sig}." if sig else ""))


def _i_operator(m, t):
    return (" A tile is not a number, so Python operators like += do not work on one. Accumulate by "
            "allocating a PSUM tile with nl.ndarray(shape, nl.float32, buffer=nl.psum) and letting "
            "nisa.nc_matmul add into it across the loop, or combine two tiles with a nisa op rather "
            "than a Python operator.")


def _i_dma_count(m, t):
    src, dst = int(m.group(1)), int(m.group(2))
    return (f" The tile you allocated holds {dst} elements but you copied {src} into it. "
            f"nisa.dma_copy does not slice or broadcast: allocate the destination with EXACTLY the "
            f"shape of the slice you are moving. If you want a 128x512 piece of a bigger tensor, "
            f"write t = nl.ndarray((128, 512), dtype=a.dtype, buffer=nl.sbuf) and then "
            f"nisa.dma_copy(dst=t, src=a[0:128, 0:512]) -- the slice on the right must have the "
            f"same shape as the tile on the left.")


def _i_partition(m, t):
    got, mx = int(m.group(2)), int(m.group(3))
    return (f" A tile may have at most {mx} rows, and you asked for {got}. Do not allocate one tile "
            f"for the whole tensor: loop over the partition dimension in chunks of at most {mx} with "
            f"nl.affine_range, allocate the tile inside the loop with the chunk's own size, and copy "
            f"one chunk at a time, e.g. src=a[i*{mx}:(i+1)*{mx}, :]. If a dimension is already {mx} "
            f"or smaller, use it whole -- do NOT pad it up to {mx}, that reads past the end of the "
            f"tensor. The same applies to where you write the result back.")


def _i_broadcast(m, t):
    val, dst = int(m.group(1)), int(m.group(2))
    return (f" You assigned {val} elements into a slice that holds {dst}. Assignment does not "
            f"reshape or broadcast either: the slice on the left and the value on the right must "
            f"have the SAME shape. If the value is bigger, you are writing a whole tile where a "
            f"slice belongs -- index the destination to match, e.g. out[i*128:(i+1)*128, :] = tile. "
            f"If it is smaller, you are looping over the wrong dimension.")


def _i_oob(m, t):
    dim, hi, size = m.group(1), int(m.group(3)), int(m.group(4))
    return (f" You indexed up to {hi} on dimension {dim}, which is only {size} long. Tile limits are "
            f"a MAXIMUM, not a target. Derive every bound from the tensor's own shape -- use "
            f"min(limit, size) and let the final chunk be partial -- rather than writing a fixed "
            f"number. Note the two limits differ: the partition dimension (first) allows at most "
            f"128, the free dimension allows more.")


def _i_contraction(m, t):
    k, mx = int(m.group(1)), int(m.group(2))
    return (f" The contraction dimension K is {k} and one nc_matmul can only contract {mx}. Split K "
            f"into chunks of {mx} and accumulate: allocate ONE psum tile OUTSIDE the K loop, call "
            f"nisa.nc_matmul into that same psum tile once per chunk so the partial products add up "
            f"there, and only after the loop copy it out with nisa.tensor_copy. Do not allocate a "
            f"new psum tile per chunk and do not write partial results to HBM.")


def _i_hbm_engine(m, t):
    return (f" `nisa.{m.group(1)}` only moves data between on-chip buffers, sbuf and psum. To reach "
            f"HBM -- the tensor you allocated with buffer=nl.shared_hbm and will return -- use "
            f"nisa.dma_copy instead. The usual sequence is nc_matmul into psum, tensor_copy psum to "
            f"sbuf, then dma_copy sbuf to the shared_hbm output.")


def _i_buffer(m, t):
    which, needed, got = m.groups()
    place = {"psum": "nl.psum", "sbuf": "nl.sbuf"}.get(needed, needed)
    return (f" Allocate the `{which}` tile with buffer={place} instead of nl.{got}. For "
            f"nisa.nc_matmul: dst must be in nl.psum, and stationary and moving must both be in "
            f"nl.sbuf. Copy between them with nisa.tensor_copy.")


def _i_contraction_mismatch(m, t):
    # ADDED on seat 73: the baseline's level-2 (transpose) runs reached for nc_matmul and hit this
    # with no translation at all -- the API card's matmul paragraph pulls non-matmul levels to it.
    s, mv = int(m.group(1)), int(m.group(2))
    return (f" nc_matmul contracts over the PARTITION (first) dimension, so stationary and moving "
            f"must have the same first dimension; you passed {s} and {mv}. If this operation is not "
            f"a matrix multiply, do not use nc_matmul at all: move data with nisa.dma_copy or "
            f"nisa.tensor_copy between slices instead.")


def _i_multi(m, t):
    return (" Pass every argument by keyword, e.g. nisa.nc_matmul(dst=..., stationary=..., "
            "moving=...), so none is bound twice.")


def _i_1d(m, t):
    return (" Every SBUF and PSUM tile needs two dimensions: a partition dimension first, then a "
            "free dimension. A 1-D tile is not allowed, so write nl.ndarray((rows, cols), ...) and "
            "give a length-N vector the shape (1, N) or (N, 1) depending on which axis you are "
            "reducing over.")


def _i_reshape(m, t):
    return (" Do not reshape. Work with the shapes you were given and slice them into tiles, e.g. "
            "src=a[0:128, 0:64].")


def _i_mod_attr(m, t):
    return _available_names(f"{m.group(1)}.{m.group(2)}")


def _i_obj_attr(m, t):
    return (f" A {m.group(1)} is not a numpy array, so it has no `{m.group(2)}`. Use the nl/nisa "
            f"functions instead.")


RULES = [
    ("call_memory_region", r"'MemoryRegion' object is not callable", _i_memregion),
    ("invented_kwarg", r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", _i_kwarg),
    ("python_operator_on_tile", r"unsupported operand type\(s\) for.*NkiTensor", _i_operator),
    ("dma_shape_mismatch", r"dma_copy requires src and dst to have the same number of elements, "
                           r"got src=(\d+), dst=(\d+)", _i_dma_count),
    ("partition_over_128", r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)",
     _i_partition),
    ("assign_shape_mismatch", r"value array of shape \((\d+),?\) could not be broadcast to "
                              r"indexing result of shape \((\d+),?\)", _i_broadcast),
    ("out_of_bounds", r"Out-of-bound access for tensor .*? on dimension (\d+): "
                      r"index range \[(\d+), (\d+)\] exceed dimension size of (\d+)", _i_oob),
    ("contraction_over_128", r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)",
     _i_contraction),
    ("engine_cannot_reach_hbm", r"(\w+) (?:dst|src)? ?must be in \['sbuf', 'psum'\], got shared_hbm",
     _i_hbm_engine),
    ("wrong_buffer", r"(\w+) must be in \['(\w+)'\], got (\w+)", _i_buffer),
    ("contraction_mismatch", r"Matmul contraction dimension mismatch: stationary\[0\]=(\d+) != "
                             r"moving\[0\]=(\d+)", _i_contraction_mismatch),
    ("arg_bound_twice", r"got multiple values for argument", _i_multi),
    ("tile_1d", r"must have at least 2 dimensions", _i_1d),
    ("reshape_instead_of_slice", r"cannot reshape array of size", _i_reshape),
    ("invented_api", r"module '([\w.]+)' has no attribute '(\w+)'", _i_mod_attr),
    ("numpy_method_on_tile", r"'(\w+)' object has no attribute '(\w+)'", _i_obj_attr),
]
_COMPILED = [(b, re.compile(p, re.S), f) for b, p, f in RULES]


def translate(error_text):
    """(bucket, error_text + instruction). Unknown errors pass through with bucket 'raised_other'."""
    for bucket, rx, build in _COMPILED:
        m = rx.search(error_text)
        if m:
            return bucket, error_text + build(m, error_text)
    return "raised_other", error_text


def enrich(error_text):
    """Drop-in for upstream agent.enrich()."""
    return translate(error_text)[1]


# ---------------------------------------------------------------- any feedback -> bucket
#
# agent.grade() produces feedback from several stages. classify() maps the WHOLE string to one
# bucket so the taxonomy can count it. Stage prefixes first, then the raw-error table.

_STAGES = [
    ("no_code", r"^No code came back"),
    ("syntax_error", r"^The code does not parse"),
    ("rule_wrong_entry_name", r"^Rule violations.*no function named"),
    ("rule_missing_jit", r"^Rule violations.*not decorated with `@nki\.jit`"),
    ("rule_partition_over_128", r"^Rule violations.*partition dimension \d+ exceeds"),
    ("rule_framework_call", r"^Rule violations.*(calls `|uses `|operator)"),
    ("rule_other", r"^Rule violations"),
    ("invented_import", r"^There is no module named"),
    ("load_error", r"could not be loaded"),
    ("cannot_simulate", r"^CANNOT SIMULATE"),
    ("solved", r"^Correct on every shape"),
]
_NUMERIC = [
    ("hardware_hazard", r"CORRECT ON CPU BUT WRONG ON HARDWARE"),
    ("traffic_over_bar", r"TOO MUCH HBM TRAFFIC"),
    ("mutates_input", r"(?i)input.*(modified|overwr|changed)"),
    ("wrong_output_shape", r"WRONG SHAPE"),
    ("non_finite_output", r"NON-FINITE OUTPUT"),
    ("output_not_written", r"OUTPUT IS \d+% ZEROS"),
    ("partial_coverage", r"% of the output is zero, in the block"),
    ("ragged_edge", r"NUMERICAL MISMATCH.*ragged edge is the likely"),
    ("wrong_arithmetic", r"NUMERICAL MISMATCH"),
]


def classify(feedback):
    fb = feedback or ""
    for bucket, pat in _STAGES:
        if re.search(pat, fb, re.S):
            return bucket
    m = re.search(r"raised (\w+): (.*)", fb, re.S)
    if m:
        return translate(m.group(0))[0]
    for bucket, pat in _NUMERIC:
        if re.search(pat, fb, re.S):
            return bucket
    return "other"


# ---------------------------------------------------------------- selftest

_CASES = [
    ("TypeError: 'MemoryRegion' object is not callable", "call_memory_region"),
    ("TypeError: nc_matmul() got an unexpected keyword argument 'transpose_moving'", "invented_kwarg"),
    ("TypeError: unsupported operand type(s) for +=: 'NkiTensor' and 'NkiTensor'",
     "python_operator_on_tile"),
    ("dma_copy requires src and dst to have the same number of elements, got src=4096, dst=128",
     "dma_shape_mismatch"),
    ("AssertionError: dma_copy dst partition dimension 256 exceeds maximum 128", "partition_over_128"),
    ("Out-of-bound access for tensor a on dimension 0: index range [0, 127] exceed dimension size "
     "of 32", "out_of_bounds"),
    ("Matmul contraction dimension 256 exceeds pmax=128", "contraction_over_128"),
    ("tensor_copy dst must be in ['sbuf', 'psum'], got shared_hbm", "engine_cannot_reach_hbm"),
    ("AssertionError: dst must be in ['psum'], got sbuf", "wrong_buffer"),
    ("SBUF and PSUM tensors must have at least 2 dimensions", "tile_1d"),
    ("ValueError: cannot reshape array of size 32768 into shape (1,64)", "reshape_instead_of_slice"),
    ("AttributeError: module 'nki.language' has no attribute 'dot'", "invented_api"),
    ("AttributeError: 'NkiTensor' object has no attribute 'mean'", "numpy_method_on_tile"),
    ("AssertionError: Matmul contraction dimension mismatch: stationary[0]=3 != moving[0]=4",
     "contraction_mismatch"),
    ("ZeroDivisionError: division by zero", "raised_other"),
]
_FEEDBACK = [
    ("No code came back. Reply with one python code block", "no_code"),
    ("The code does not parse: invalid syntax on line 3.", "syntax_error"),
    ("Rule violations, which score zero however fast the kernel is. Fix exactly these: no function "
     "named `x` is defined", "rule_wrong_entry_name"),
    ("Rule violations, which score zero however fast the kernel is. Fix exactly these: `k` is not "
     "decorated with `@nki.jit`, so it", "rule_missing_jit"),
    ("0 of 4 shapes passed. On C,H,W=(32, 32, 32) pool=2: raised AssertionError: dst must be in "
     "['psum'], got sbuf Allocate", "wrong_buffer"),
    ("1 of 4 shapes passed. On K=256: NUMERICAL MISMATCH: worst ... this is in the final partial "
     "PARTITION tile (partition 3 of 4) -- the ragged edge is the likely cause", "ragged_edge"),
    ("1 of 4 shapes passed. On K=256: NUMERICAL MISMATCH: worst ... most elements are wrong",
     "wrong_arithmetic"),
    ("3 of 4 shapes passed. On K=256 M=512 N=1024: CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS "
     "LEVEL", "traffic_over_bar"),
    ("Correct on every shape.", "solved"),
]


def _sdk():
    try:
        import nki  # noqa: F401
        return True
    except ImportError:
        return False


def selftest():
    bad = 0
    for err, want in _CASES:
        got, text = translate(err)
        # invented_api lists real names from the SDK, so it can only add text where nki imports.
        needs_sdk = want == "invented_api" and not _sdk()
        ok = got == want and (want == "raised_other" or needs_sdk or len(text) > len(err))
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} translate -> {got:28s} {err[:60]}")
    for fb, want in _FEEDBACK:
        got = classify(fb)
        bad += got != want
        print(f"  {'ok  ' if got == want else 'FAIL'} classify  -> {got:28s} {fb[:60]}")
    print("VERDICTS SELFTEST " + ("PASSED" if not bad else f"FAILED ({bad})"))
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if selftest() else 0)
