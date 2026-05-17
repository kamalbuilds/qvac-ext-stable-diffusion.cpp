# LTX-2.3 Video Generation (T2V + I2V)

This document describes the LTX-2.3 video generation support added to this fork
(`qvac-ext-stable-diffusion.cpp`) and the companion Bare addon for the QVAC ecosystem.
It covers the model, the conversion pipeline, build and usage, quantization guidance,
and the benchmark methodology.

## Status

| Area | State |
| --- | --- |
| Safetensors to GGUF conversion (f16, q4_0, q5_1, q8_0) | implemented, self-tested against the real `ltx-2.3-22b-dev` header |
| Architecture detection and model load on CPU | in progress |
| End-to-end T2V and I2V on CPU (AVX, NEON) | planned (M2) |
| Vulkan and Metal backends + benchmarks | planned (M3) |
| `ltx2.h` C API + Bare addon + tests + docs | planned (M4) |

## Model

LTX-2.3 is an open-weights video diffusion model from Lightricks. The published checkpoint
`Lightricks/LTX-2.3` ships `ltx-2.3-22b-dev.safetensors` (a 22B-parameter transformer with a
combined video and audio stream) plus distilled and upscaler variants. This work targets the
**video stream only** for text-to-video (T2V) and image-to-video (I2V). The audio DiT, Audio-VAE,
vocoder, audio-to-video cross-attention, the spatial and temporal upscalers, and training are out
of scope.

Core components used here:

- **Video DiT.** Patchify projection, AdaLN single modulation, a learnable
  `video_embeddings_connector` (1D self-attention with register tokens), and the stack of
  transformer blocks (self-attention `attn1`, cross-attention `attn2` to the text context, and a
  gated feed-forward).
- **Video-VAE.** Spatiotemporal autoencoder with 32x32x8 compression per token, with
  `per_channel_statistics` (mean-of-means, std-of-means) normalization. The encoder is used for
  I2V conditioning; the decoder turns final latents into RGB frames.
- **Gemma 3 text encoder.** Produces the conditioning context, projected into the DiT cross-attention
  dimension.
- **Scheduler.** Flow-matching denoiser (`LTX2Scheduler`) with classifier-free guidance.

## Download weights

- **LTX-2.3 diffusion model**
  - safetensors: https://huggingface.co/Kijai/LTX2.3_comfy/tree/main/diffusion_models
  - reference GGUF: https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main
- **Gemma-3-12b-it text encoder**: https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/tree/main
- **Embeddings connectors**: https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/text_encoders
- **Video VAE**: https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/vae

The audio VAE is published in the same folder but is not used by this video-only build.

## Conversion: `script/convert_ltx2.py`

Converts an LTX-2.3 safetensors checkpoint to a single GGUF file for ggml inference.

```bash
pip install -r script/requirements-ltx2.txt
# Inspect a checkpoint without writing output:
python3 script/convert_ltx2.py ltx-2.3-22b-dev.safetensors --dry-run
# Convert at Q8_0, bundling the Video-VAE:
python3 script/convert_ltx2.py ltx-2.3-22b-dev.safetensors --vae ltx-vae.safetensors \
    --quant q8_0 -o ltx2-22b-q8_0.gguf
# Stdlib-only correctness check used by CI (no weights required):
python3 script/convert_ltx2.py --self-test
```

Design points:

- **Video-stream only.** Every audio-coupled tensor is dropped (audio DiT, Audio-VAE, vocoder, and
  both sides of the audio-to-video cross-attention). The `--self-test` asserts zero audio tensors
  survive against the real checkpoint header.
- **Selective precision.** Norms, biases, embeddings, modulation tables, the patchify and output
  projections, the connector, and the VAE per-channel statistics are always kept at F16 or higher,
  even when the body of the model is quantized.
- **Validation.** Each tensor's declared buffer length is checked against shape and dtype before it
  is accepted, so a malformed or truncated checkpoint fails fast.

## Build

LTX-2.3 inference relies on 3D convolution ops in the bundled ggml. Build as usual for this repo,
then enable the backend appropriate to your platform (CPU AVX/AVX2/AVX512 on x86-64, CPU NEON on
ARM64, Metal on macOS, Vulkan on Linux and Windows). See `docs/build.md` for the base build.

## Usage (CLI)

### LTX-2.3 dev T2V

```
sd-cli -M vid_gen \
    --diffusion-model       ltx-2.3-22b-dev-Q4_K_M.gguf \
    --vae                   ltx-2.3-22b-dev_video_vae.safetensors \
    --llm                   gemma-3-12b-it-Q4_K.gguf \
    --embeddings-connectors ltx-2.3-22b-dev_embeddings_connectors.safetensors \
    -p "a lovely cat" \
    -n "worst quality, low quality, blurry, distorted, artifacts" \
    --cfg-scale 6.0 --sampling-method euler -v \
    -W 1280 -H 720 --video-frames 33 --fps 24 \
    --diffusion-fa --offload-to-cpu \
    -o t2v.webm
```

### LTX-2.3 dev I2V

```
sd-cli -M vid_gen \
    --diffusion-model       ltx-2.3-22b-dev-Q4_K_M.gguf \
    --vae                   ltx-2.3-22b-dev_video_vae.safetensors \
    --llm                   gemma-3-12b-it-Q4_K.gguf \
    --embeddings-connectors ltx-2.3-22b-dev_embeddings_connectors.safetensors \
    -p "a lovely cat" --cfg-scale 6.0 --sampling-method euler -v \
    -W 1280 -H 720 --video-frames 33 \
    --diffusion-fa --offload-to-cpu \
    -i reference.png \
    -o i2v.webm
```

### LTX-2.3 dev FLF2V (first-last frame to video)

```
sd-cli -M vid_gen \
    --diffusion-model       ltx-2.3-22b-dev-Q4_K_M.gguf \
    --vae                   ltx-2.3-22b-dev_video_vae.safetensors \
    --llm                   gemma-3-12b-it-Q4_K.gguf \
    --embeddings-connectors ltx-2.3-22b-dev_embeddings_connectors.safetensors \
    -p "glass flower blossom" --cfg-scale 6.0 --sampling-method euler -v \
    -W 1280 -H 720 --video-frames 33 \
    --diffusion-fa --offload-to-cpu \
    --init-img start_image.png \
    --end-img   end_image.png \
    -o flf2v.webm
```

## Quantization guidance

| Level | Use case | Notes |
| --- | --- | --- |
| `f16` | Reference quality, benchmarks | Largest footprint; used for PSNR/SSIM correctness checks against PyTorch |
| `q8_0` | Best quality per byte | Recommended default for capable machines |
| `q5_1` | Balanced | Good quality with reduced memory |
| `q4_0` | Constrained devices | Target peak memory under 12 GB RAM (CPU) or 10 GB VRAM (GPU) |

The VAE is sensitive to precision. On CPU, quantized 3D convolutions take an explicit
im2col plus matmul path to avoid an F16-only assertion in the ggml CPU backend, and the most
overflow-prone encoder stage is forced to F32.

## Benchmark methodology

The acceptance target is a wall-clock speed-up over the PyTorch and Diffusers pipeline at F16 on
identical hardware. Reported with a fixed seed, fixed resolution and frame count, a warm cache, and
the median of three runs:

- Tokens per second, total generation time, and peak memory, for at least two quantization levels
  per backend.
- Output quality: PSNR and SSIM of this implementation against the PyTorch reference at F16, for
  both T2V and I2V.
- On macOS Metal, Draw Things is used as the on-device performance baseline.
