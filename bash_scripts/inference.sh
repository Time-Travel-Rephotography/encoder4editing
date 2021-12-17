set -exo

list="$1"
ckpt="${2:-pretrained_models/e4e_ffhq_encode.pt}"

base_dir="$REPHOTO/dataset/historically_interesting/aligned/manual_celebrity_in_19th_century/tier1/${list}/"
save_dir="results_test/${list}/"


TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
PYTHONPATH="" \
python scripts/inference.py \
    --images_dir="${base_dir}" \
    --save_dir="${save_dir}" \
    "${ckpt}"
