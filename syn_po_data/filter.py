import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import re
import json
import random
import string
import argparse
from transformers import AutoTokenizer
from template import PROMPT_DICT
from collections import Counter


random.seed(123)  # 你可以选择任何整数作为种子


ans_prefixes = [
    "answer:",
    "the answer is:",
    "final answer is:",
]

def extract_answer(pred):
    pred = pred.lower()
    flag = False
    for prefix in ans_prefixes:
        idx = pred.rfind(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            break
        ans = pred[idx + len(prefix):]
        flag = True
        return ans.strip(), flag
    return pred.strip(), flag

# 读取 JSONL 文件内容为列表
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def substring_exact_match_score(prediction, ground_truth):
    """Check if the ground truth is a (soft) exact match substring of the prediction."""
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    for truth in ground_truth:
        norm_truth = normalize_answer(truth)
        norm_prediction = normalize_answer(prediction)
        if norm_truth in norm_prediction:
            return 1.0
    return 0.0

def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def F1_scorer(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, qa_f1_score(prediction, ground_truth))
    return score


def evaluate_output(output, ground_truth):
    pred_ans, flag = extract_answer(output)
    f1_score = 0
    subem_score = 0
    if flag:
        # 符合格式才计算奖励，否则为0
        f1_score = F1_scorer(pred_ans, ground_truth)
        subem_score = substring_exact_match_score(pred_ans, ground_truth)
    return (subem_score + f1_score) / 2.0


def filter_metric(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 加载输入数据
    with open(args.mab_input_dir, 'r', encoding='utf-8') as file:
        mab_data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
    print(len(mab_data))

    os.makedirs(os.path.dirname(args.dpo_dir), exist_ok=True)
    # os.makedirs(os.path.dirname(args.sft_dir), exist_ok=True)

    cut_off_cnt = 0
    final_nums = 0

    # sft_f = open(args.sft_dir, 'w')
    dpo_f = open(args.dpo_dir, 'w')

    # 通过metric挑选偏好数据
    for index, example in enumerate(mab_data):
        if index % 50 == 0:
            print(f"processing example #{index}")
            print(final_nums)
        ground_truth = example["answer"]
        all_predictions = []

        # 添加 mab cot
        if "samples" in example:
            mab_preds = [example["samples"][rollout_idx]["pred"] for rollout_idx in range(args.sample_nums-2)]
            all_predictions.extend(mab_preds)
        elif "pred" in example:
            mab_preds = example["pred"]
            all_predictions.extend(mab_preds)
        else:
            raise NotImplementedError("mab error!")

        # 计算分数
        scores = [evaluate_output(pred, ground_truth) for pred in all_predictions]

        # 找出最大和最小分数
        max_score = max(scores)
        min_score = min(scores)

        if max_score <= 0.5:
            continue
        # if max_score - min_score < 0.5:
        #     continue

        if max_score == min_score:
            continue

        # 找出所有最大/最小得分对应的索引
        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        min_indices = [i for i, s in enumerate(scores) if s == min_score]

        random.shuffle(max_indices)
        random.shuffle(min_indices)

        # 构造偏好response
        chosen = all_predictions[max_indices[0]]
        rejected = all_predictions[min_indices[0]]

        query = example['input']
        token_query = tokenizer([query])
        query_length = len(token_query.input_ids[0])

        chosen_len = len(tokenizer([chosen]).input_ids[0])
        rejected_len = len(tokenizer([rejected]).input_ids[0])
        max_response_len = max(chosen_len, rejected_len)

        context = example['context']
        token_context = tokenizer([context]).input_ids[0]
        context_len = len(token_context)

        budget = args.max_prompt_length - 32 - query_length - max_response_len
        if context_len > budget:
            cut_off_cnt += 1
            split = budget // 2
            token_context = token_context[0:split] + token_context[-split:]

        new_context = tokenizer.decode(token_context, skip_special_tokens=True)
        input_data = PROMPT_DICT['cot_answer'].format(context=new_context, question=query)

        if index == 0:
            message = [
                {"role": "user", "content": input_data},
            ]
            prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            print(prompt)

        dpo_example = {
            "id": example["id"],
            "input": example["input"],
            "answer": example["answer"],
            "prompt": input_data,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_reward": max_score,
            "rejected_reward": min_score,
            "chosen_index": max_indices[0],
            "rejected_index": min_indices[0],
        }
        dpo_f.write(json.dumps(dpo_example, ensure_ascii=False) + "\n")

        # sft_example = {
        #     "id": example["id"],
        #     "input": example["input"],
        #     "answer": example["answer"],
        #     "prompt": input_data,
        #     "output": chosen,
        #     "chosen_reward": max_score,
        #     "chosen_index": max_indices[0],
        # }
        # sft_f.write(json.dumps(sft_example, ensure_ascii=False) + "\n")
        final_nums += 1
        if final_nums >= 2250:
            break

    print(cut_off_cnt)
    print(final_nums)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("---model_path", type=str, default="/data1/duanshaohua/hf_hub/models--Qwen--Qwen2.5-7B-Instruct")
    parser.add_argument("---model_path", type=str, default="/data1/duanshaohua/hf_hub/models--meta-llama--Llama-3.1-8B-Instruct")
    parser.add_argument('--mab_input_dir', type=str, default="/data1/duanshaohua/rankcot/data/Llama3.1/mab/2048/top4/step2_rollout_top4_0_5000.jsonl")
    parser.add_argument('--dpo_dir', type=str, default="/data1/duanshaohua/rankcot/data/Llama3.1/mab/2048/top4/llama_longmab_dpodata_2k_0_5000_filter.jsonl")
    # parser.add_argument('--sft_dir', type=str, default="/data1/duanshaohua/rankcot/data/Llama3.1/mab/2048/top4/minicpm_embedding/llama3_minicpm_embedding_top4_samples32_sftdata_prompt8k_0_5000.jsonl")
    parser.add_argument('--sample_nums', type=str, default=32)
    parser.add_argument('--max_prompt_length', type=int, default=8 * 1024)

    args = parser.parse_args()
    filter_metric(args)
