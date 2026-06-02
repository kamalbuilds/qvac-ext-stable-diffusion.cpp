#!/bin/zsh
# Real LTX-2.3 generation smoke on the Mac mini (M4, 16 GB).
# Distilled model + few steps + tiny resolution so it fits 16 GB and finishes.
# Proves the full pipeline: GGUF load -> Gemma encode -> LTX-2 DiT denoise ->
# Video-VAE decode -> frame output, on real weights.
set -e
M=~/ltx2/models
B=~/ltx2/fork/build/bin/sd-cli
OUT=~/ltx2/out
mkdir -p "$OUT"

DIFF="$M/ltx-2.3-22b-distilled-1.1-Q2_K.gguf"
VAE="$M/ltx-2.3-22b-dev_video_vae.safetensors"
CONN="$M/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
LLM="$M/gemma-3-12b-it-Q2_K.gguf"

for f in "$DIFF" "$VAE" "$CONN" "$LLM"; do
  if [ ! -f "$f" ]; then echo "MISSING: $f"; exit 2; fi
done

echo "=== [$(date)] LTX-2.3 distilled T2V smoke (256x256, 9 frames, 8 steps) ==="
"$B" -M vid_gen \
  --diffusion-model "$DIFF" \
  --vae "$VAE" \
  --embeddings-connectors "$CONN" \
  --llm "$LLM" \
  -p "a lovely cat playing with a ball of yarn" \
  -n "worst quality, blurry, distorted" \
  --cfg-scale 1.0 \
  --sampling-method euler \
  --steps 8 \
  -W 256 -H 256 \
  --video-frames 9 --fps 24 \
  --diffusion-fa \
  --offload-to-cpu \
  -v \
  -o "$OUT/ltx2_cat_%03d.png"

echo "=== [$(date)] output frames ==="
ls -lh "$OUT"/*.png 2>/dev/null
echo "GEN_DONE"
