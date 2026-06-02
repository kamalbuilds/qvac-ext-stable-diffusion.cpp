#!/usr/bin/env python3
"""Convert an LTX-2.3 safetensors checkpoint to GGUF for ggml inference.

This converter targets the *video stream only* (text-to-video and image-to-video).
Every audio-coupled tensor (audio DiT, Audio-VAE, vocoder, and both sides of the
audio-to-video cross-attention) is dropped, since audio is out of scope for this
fork. Norms, biases, embeddings, modulation tables, the patchify and output
projections, the learnable connector, and the VAE per-channel statistics are kept
at high precision even when the body of the model is quantized.

Examples
--------
Inspect a checkpoint without writing output (no heavy dependencies required):
    python3 script/convert_ltx2.py ltx-2.3-22b-dev.safetensors --dry-run

Convert at Q8_0, bundling the Video-VAE into the same GGUF:
    python3 script/convert_ltx2.py ltx-2.3-22b-dev.safetensors \\
        --vae ltx-vae.safetensors --quant q8_0 -o ltx2-22b-q8_0.gguf

Run the stdlib-only correctness check used by CI (no weights required):
    python3 script/convert_ltx2.py --self-test
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import struct
import sys
from typing import Dict, List, Tuple

LOG = logging.getLogger("convert_ltx2")

# --- Tensor classification -------------------------------------------------

# Any tensor whose name contains one of these substrings belongs to the audio
# path or the audio<->video cross-attention and is dropped (video-only build).
DROP_SUBSTRINGS: Tuple[str, ...] = (
    "audio",          # audio DiT blocks, audio tables, text_encoders audio, etc.
    "vocoder",        # vocoder weights
    "audio_vae",      # Audio-VAE
    "_a2v_",          # audio-to-video cross-attention (covers a2v_ca_audio/video)
    "_v2a_",          # video-to-audio cross-attention
)

# Tensors that must never be quantized below F16, regardless of --quant. These
# are small, precision-sensitive, and cheap to keep at full width.
KEEP_HIGH_PRECISION_SUBSTRINGS: Tuple[str, ...] = (
    ".bias",
    "norm",                       # layer / rms / group norms
    "embed",                      # embeddings
    "scale_shift_table",          # AdaLN modulation tables
    "patchify_proj",
    "proj_out",
    "adaln_single",
    "prompt_adaln_single",
    "video_embeddings_connector",
    "per_channel_statistics",     # VAE normalization stats
)

SUPPORTED_QUANTS = ("f16", "q4_0", "q5_1", "q8_0")

DEFAULT_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "testdata", "ltx23_tensor_names.json.gz"
)


def is_audio_tensor(name: str) -> bool:
    lname = name.lower()
    return any(sub in lname for sub in DROP_SUBSTRINGS)


def keep_high_precision(name: str) -> bool:
    return any(sub in name for sub in KEEP_HIGH_PRECISION_SUBSTRINGS)


def classify(names: List[str]) -> Tuple[List[str], List[str]]:
    """Split tensor names into (kept_video, dropped_audio)."""
    kept: List[str] = []
    dropped: List[str] = []
    for name in names:
        (dropped if is_audio_tensor(name) else kept).append(name)
    return kept, dropped


# --- safetensors header (stdlib only) --------------------------------------

def read_safetensors_header(path: str) -> Dict[str, dict]:
    """Read only the JSON header of a .safetensors file. No tensor data loaded."""
    with open(path, "rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: file too short to be a safetensors checkpoint")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 0 or header_len > 200_000_000:
            raise ValueError(f"{path}: implausible header length {header_len}")
        header = json.loads(fh.read(header_len).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


_DTYPE_SIZE = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def validate_entry(name: str, entry: dict) -> None:
    """Check that declared byte range matches shape * dtype size."""
    dtype = entry["dtype"]
    shape = entry["shape"]
    begin, end = entry["data_offsets"]
    if dtype not in _DTYPE_SIZE:
        raise ValueError(f"{name}: unknown dtype {dtype}")
    numel = 1
    for d in shape:
        numel *= d
    expected = numel * _DTYPE_SIZE[dtype]
    actual = end - begin
    if expected != actual:
        raise ValueError(
            f"{name}: buffer/shape mismatch (shape={shape} dtype={dtype} "
            f"expected {expected} bytes, header declares {actual})"
        )


# --- Commands --------------------------------------------------------------

def cmd_dry_run(args: argparse.Namespace) -> int:
    header = read_safetensors_header(args.checkpoint)
    names = list(header.keys())
    for name in names:
        validate_entry(name, header[name])
    kept, dropped = classify(names)
    hp = sum(1 for n in kept if keep_high_precision(n))
    LOG.info("checkpoint: %s", args.checkpoint)
    LOG.info("total tensors:       %d", len(names))
    LOG.info("kept (video stream): %d", len(kept))
    LOG.info("dropped (audio):     %d", len(dropped))
    LOG.info("kept at high prec.:  %d  (quantized: %d)", hp, len(kept) - hp)
    LOG.info("would quantize to:   %s  (use without --dry-run to write GGUF)", args.quant)
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    """Stdlib-only correctness check against the bundled tensor-name fixture."""
    fixture = args.fixture or DEFAULT_FIXTURE
    if not os.path.exists(fixture):
        LOG.error("fixture not found: %s", fixture)
        return 2
    with gzip.open(fixture, "rt") as fh:
        names = json.load(fh)

    kept, dropped = classify(names)
    audio_leaks = [n for n in kept if is_audio_tensor(n)]

    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        ok = ok and cond
        LOG.info("  [%s] %s%s", status, label, f" ({detail})" if detail else "")

    LOG.info("LTX-2.3 converter self-test against %s", os.path.basename(fixture))
    check("total input tensors == 5947", len(names) == 5947, f"got {len(names)}")
    check("audio tensors are dropped", len(dropped) > 0, f"dropped {len(dropped)}")
    check("no audio tensor leaks into kept set", not audio_leaks,
          f"{len(audio_leaks)} leaks")
    check("kept video tensor count in [1750,1770]", 1750 <= len(kept) <= 1770,
          f"kept {len(kept)}")

    LOG.info("self-test: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _target_qtype(name: str, quant: str):
    """Return the gguf quantization type for a tensor given --quant."""
    import gguf  # lazy: heavy dependency only needed for real conversion

    if quant == "f16" or keep_high_precision(name):
        return gguf.GGMLQuantizationType.F16
    return {
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "q5_1": gguf.GGMLQuantizationType.Q5_1,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
    }[quant]


def cmd_convert(args: argparse.Namespace) -> int:
    import numpy as np  # noqa: F401  (used via safetensors / gguf)
    import gguf
    from safetensors import safe_open

    out_path = args.output or _default_output(args.checkpoint, args.quant)
    writer = gguf.GGUFWriter(out_path, arch="ltx2")
    writer.add_name("LTX-2.3")
    writer.add_description("LTX-2.3 video diffusion (video stream only)")
    writer.add_file_type({
        "f16": gguf.GGMLQuantizationType.F16,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "q5_1": gguf.GGMLQuantizationType.Q5_1,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
    }[args.quant])

    written = 0
    sources = [args.checkpoint] + ([args.vae] if args.vae else [])
    for src in sources:
        header = read_safetensors_header(src)
        for name in header:
            validate_entry(name, header[name])
        with safe_open(src, framework="numpy") as fh:
            for name in fh.keys():
                if is_audio_tensor(name):
                    continue
                data = fh.get_tensor(name)
                if data.dtype == np.dtype("bfloat16") if hasattr(np, "bfloat16") else False:
                    data = data.astype(np.float32)
                qtype = _target_qtype(name, args.quant)
                if qtype == gguf.GGMLQuantizationType.F16:
                    out = data.astype(np.float16)
                    writer.add_tensor(name, out, raw_dtype=qtype)
                else:
                    quantized = gguf.quants.quantize(data.astype(np.float32), qtype)
                    writer.add_tensor(name, quantized, raw_dtype=qtype,
                                      raw_shape=list(data.shape))
                written += 1
                if written % 200 == 0:
                    LOG.info("  ... %d tensors written", written)

    LOG.info("writing %s (%d tensors, quant=%s)", out_path, written, args.quant)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    LOG.info("done: %s", out_path)
    return 0


def _default_output(checkpoint: str, quant: str) -> str:
    base = os.path.splitext(os.path.basename(checkpoint))[0]
    return f"{base}-{quant}.gguf"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_ltx2.py",
        description="Convert LTX-2.3 safetensors (video stream) to GGUF.",
    )
    p.add_argument("checkpoint", nargs="?",
                   help="path to the LTX-2.3 diffusion safetensors file")
    p.add_argument("--vae", help="optional Video-VAE safetensors to bundle into the GGUF")
    p.add_argument("-o", "--output", help="output GGUF path (default: <name>-<quant>.gguf)")
    p.add_argument("--quant", choices=SUPPORTED_QUANTS, default="q8_0",
                   help="quantization level for the model body (default: q8_0)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse and validate the header, print kept/dropped counts, write nothing")
    p.add_argument("--self-test", action="store_true",
                   help="run the stdlib-only correctness check (no weights needed)")
    p.add_argument("--fixture", help="override the self-test tensor-name fixture path")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if args.self_test:
        return cmd_self_test(args)
    if not args.checkpoint:
        LOG.error("a checkpoint path is required (or use --self-test)")
        return 2
    if not os.path.exists(args.checkpoint):
        LOG.error("checkpoint not found: %s", args.checkpoint)
        return 2
    if args.dry_run:
        return cmd_dry_run(args)
    return cmd_convert(args)


if __name__ == "__main__":
    sys.exit(main())
