# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0

# API_CONFIG=src/configs/openai_api_key_config.json
# API_CONFIG=src/configs/qwen_vl_api_key_config.json
# API_CONFIG=src/configs/internvl_api_key_config.json
# API_CONFIG=src/configs/claude_api_key_config.json
# API_CONFIG=src/configs/minicpm_api_key_config.json
# API_CONFIG=src/configs/llama_api_key_config.json
# API_CONFIG=src/configs/phi_api_key_config.json
API_CONFIG=src/configs/gemini_api_key_config.json

DATA_PATHS=(
  "data/HumanEval-V/HumanEval.jsonl"
  "data/Algorithm/Algorithm.jsonl"
  "data/MATH/MATH.jsonl"
)

IMAGE_DIRS=(
  "data/HumanEval-V/images"
  "data/Algorithm/images"
  "data/MATH/images"
)

OUTPUT_DIR=output_text_only
mkdir -p $OUTPUT_DIR

for i in "${!DATA_PATHS[@]}"; do
  DATA_PATH=${DATA_PATHS[$i]}
  IMAGE_DIR=${IMAGE_DIRS[$i]}

  echo "Running evaluation with DATA_PATH: $DATA_PATH and IMAGE_DIR: $IMAGE_DIR"

  python3 src/evaluation/evaluate.py \
      --api_config $API_CONFIG \
      --data_path $DATA_PATH \
      --image_dir $IMAGE_DIR \
      --output_dir $OUTPUT_DIR
done