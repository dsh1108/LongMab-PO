export CUDA_VISIBLE_DEVICES=0
export INPUT_DIR="/data1/duanshaohua/hf_hub/datasets--THUDM--LongBench/processed/"
export TEMPERATURE=0.0
export BATCH_SIZE=10
export TEMPLATE="cot_answer"

# 多个模型路径
MODEL_PATHS=(
#  "/data1/duanshaohua/hf_hub/models--meta-llama--Llama-3.1-8B-Instruct"
  "/data1/duanshaohua/rankcot/checkpoint/llamafactory/Qwen2.5/new_qwen_full_doc_dpo_aligned/new_qwen_full_doc_dpo_600_aligned"
#  "/data1/duanshaohua/rankcot/checkpoint/llamafactory/Llama3.1/mab/top3/samples32/prompt8k_dpo_llama3.1/llama3_longmab_top3_4000"
#  "/data1/duanshaohua/rankcot/checkpoint/llamafactory/Llama3.1/mab/top5/samples32/prompt8k_dpo_llama3.1/llama3_longmab_top5_4000"

)

# 对应的输出路径
OUTPUT_DIRS=(
  "/data1/duanshaohua/rankcot/result/Qwen2.5/new_qwen_full_doc_dpo_600_aligned"
#  "/data1/duanshaohua/rankcot/result/Llama3.1/mab/top3/llama3_longmab_top3_4000"
#  "/data1/duanshaohua/rankcot/result/Llama3.1/mab/top5/llama3_longmab_top5_4000"
)


# 获取数组长度
num_models=${#MODEL_PATHS[@]}

for (( i=0; i<$num_models; i++ ))
do
  export MODEL_PATH=${MODEL_PATHS[$i]}
  export OUTPUT_DIR=${OUTPUT_DIRS[$i]}

  echo "Running evaluation for model: $MODEL_PATH"
  echo "Output will be saved to: $OUTPUT_DIR"

  python eval_lb_v1.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --input_dir $INPUT_DIR \
    --temperature $TEMPERATURE \
    --batch_size $BATCH_SIZE \
    --template $TEMPLATE


  echo "Finished model $i"
  echo "--------------------------"
done
