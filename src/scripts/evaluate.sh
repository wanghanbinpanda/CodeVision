export CUDA_VISIBLE_DEVICES=4,5,6,7
# API_CONFIG=src/configs/openai_api_key_config.json
#API_CONFIG=src/configs/qwen_vl_api_key_config.json
#API_CONFIG=src/configs/internvl_api_key_config.json
# API_CONFIG=src/configs/claude_api_key_config.json
# API_CONFIG=src/configs/minicpm_api_key_config.json
# API_CONFIG=src/configs/llama_api_key_config.json
API_CONFIG=src/configs/gemini_api_key_config.json

DATA_PATH=data/HumanEval-V/HumanEval.jsonl
# DATA_PATH=data/HumanEval-V-MASK/HumanEval.jsonl
# DATA_PATH=data/Algorithm/Algorithm.jsonl
# DATA_PATH=data/MATH/MATH.jsonl



IMAGE_DIR=data/HumanEval-V/images
# IMAGE_DIR=data/HumanEval-V-MASK/images
# IMAGE_DIR=data/Algorithm/images
# IMAGE_DIR=data/MATH/images

OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR


python3 src/evaluation/evaluate.py \
    --api_config $API_CONFIG \
    --data_path $DATA_PATH \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR


