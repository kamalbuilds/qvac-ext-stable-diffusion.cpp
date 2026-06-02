#!/bin/zsh
# Download a 16GB-RAM-feasible LTX-2.3 GGUF set to the Mac mini for a real
# load + small generation test. Distilled model (few-step, coherent at low
# quant) + small Gemma so peak resident memory stays under ~10 GB with
# --offload-to-cpu. Resumable via curl -C -. Logs progress.
set -e
M=~/ltx2/models
mkdir -p "$M"
cd "$M"

dl () {
  local url="$1" out="$2"
  if [ -f "$out" ]; then echo "have $out"; return 0; fi
  echo "=== [$(date)] downloading $out"
  curl -L --fail --progress-bar -C - -o "$out.part" "$url"
  mv "$out.part" "$out"
}

BASE=https://huggingface.co/unsloth

dl "$BASE/LTX-2.3-GGUF/resolve/main/distilled-1.1/ltx-2.3-22b-distilled-1.1-Q2_K.gguf" \
   "ltx-2.3-22b-distilled-1.1-Q2_K.gguf"
dl "$BASE/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-dev_video_vae.safetensors" \
   "ltx-2.3-22b-dev_video_vae.safetensors"
dl "$BASE/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors" \
   "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
dl "$BASE/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q2_K.gguf" \
   "gemma-3-12b-it-Q2_K.gguf"

echo "=== [$(date)] all files present ==="
ls -lh "$M"
echo "DOWNLOAD_OK"
