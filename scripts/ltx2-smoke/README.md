# LTX-2.3 smoke test (Apple Silicon, 16 GB)

A reproducible end-to-end check that the LTX-2.3 video pipeline in this fork
loads real weights and runs a generation on a 16 GB Apple Silicon machine
(verified on an M4 Mac mini). It uses the **distilled** model at Q2_K plus a
small Gemma quant so peak resident memory stays under ~10 GB with
`--offload-to-cpu`.

## 1. Build sd-cli (Metal)

```bash
git submodule update --init --recursive ggml
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSD_METAL=ON -DSD_WEBP=OFF -DSD_WEBM=OFF -DBUILD_SHARED_LIBS=OFF
ninja -C build sd-cli
```

## 2. Download a 16 GB-feasible weight set (~14.5 GB)

```bash
./scripts/ltx2-smoke/download-distilled-16gb.sh
```

Fetches into `~/ltx2/models/`:
- `ltx-2.3-22b-distilled-1.1-Q2_K.gguf` (distilled DiT, few-step)
- `ltx-2.3-22b-dev_video_vae.safetensors`
- `ltx-2.3-22b-distilled_embeddings_connectors.safetensors`
- `gemma-3-12b-it-Q2_K.gguf`

## 3. Generate

```bash
./scripts/ltx2-smoke/run-smoke.sh
```

Produces a 9-frame, 256x256 clip from the prompt "a lovely cat playing with a
ball of yarn" as a PNG sequence in `~/ltx2/out/`. This exercises the full
path: GGUF load, architecture detection, Gemma 3 text encode, LTX-2 DiT
denoise with the LTX2 scheduler, and Video-VAE decode.

For a higher-quality clip on a machine with more memory, use the full
`ltx-2.3-22b-dev` weights at Q4_K_M or higher and increase resolution to
1280x704 (32-aligned), frames to a higher `8*k + 1` value, and steps to ~30.
