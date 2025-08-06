export CHUNK_SIZE=1500
export RAW_DARASET="/data1/duanshaohua/rankcot/data/musique/raw_data_42_8k_16k.jsonl"
export MODEL_PATH="/data1/duanshaohua/hf_hub/models--meta-llama--Llama-3.1-8B-Instruct"
export OUTPUT_DIR="/data1/duanshaohua/rankcot/data/Llama3.1/mab/${CHUNK_SIZE}"

start_end_pairs=(
  "0 500"
  "500 1000"
#  "1000 1500"
#  "1500 2000"
#  "2000 2500"
#  "2500 3000"
#  "3000 3500"
#  "3500 4000"
#  "4000 4500"
#  "4500 5000"
)

export CUDA_VISIBLE_DEVICES=0

[ ! -d "$OUTPUT_DIR" ] && mkdir -p "$OUTPUT_DIR"

for pair in "${start_end_pairs[@]}"; do
 read START END <<< "$pair"
 echo "Model Path $MODEL_PATH"
 echo "Running gen_probe from $START to $END"
 python gen_probe.py \
   --model_path $MODEL_PATH \
   --input_dir $RAW_DARASET \
   --output_dir "${OUTPUT_DIR}/step1_passage2probe_${START}_${END}.jsonl" \
   --chunk_size $CHUNK_SIZE \
   --start $START \
   --end $END
done
