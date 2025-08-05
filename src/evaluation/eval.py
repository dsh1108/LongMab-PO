import os
import re
import json
import string
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections import Counter
from template import PROMPT_DICT
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

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
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    prediction = normalize_answer(prediction)
    for truth in ground_truth:
        truth = normalize_answer(truth)
        if truth in prediction:
            return 1.0
    return 0.0

def drqa_exact_match_score(prediction, ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    for truth in ground_truth:
        if normalize_answer(prediction) == normalize_answer(truth):
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

class llmDataset(Dataset):
    def __init__(self, data, tokenizer, template, dataname, max_input_length):
        self.data = data
        self.tokenizer = tokenizer
        self.template = template
        self.dataname = dataname
        self.max_input_length = max_input_length

    def process_prompt(self, item):
        question = item['input']
        context = item['context']
        if args.template == "direct_answer":
            template = PROMPT_DICT[self.template][self.dataname].format(context=context, question=question)
        elif args.template == "cot_answer":
            template = PROMPT_DICT[self.template].format(context=context, question=question)
        else:
            raise NotImplementedError("template error!")

        input_ids = self.tokenizer.encode(template)
        if len(input_ids) > self.max_input_length:
            input_ids = input_ids[:self.max_input_length//2] + input_ids[-self.max_input_length//2:]
            template = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        message = [
            {"role": "user", "content": template},
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        item['input_prompt'] = prompt
        return item

    def __getitem__(self, index):
        item = self.data[index]
        item = self.process_prompt(item)

        if index == 0:
            print(item)

        return item

    def __len__(self):
        return len(self.data)

    def Collactor(self, batch):
        id = [f['id'] for f in batch]
        input = [f['input'] for f in batch]
        context = [f['context'] for f in batch]
        answers = [f['answers'] for f in batch]
        input_prompt = [f['input_prompt'] for f in batch]

        return {
            'id': id,
            'input': input,
            'context': context,
            'answers': answers,
            'input_prompt': input_prompt,
        }


def inference(args):

    # 测试集路径
    base_path = args.input_dir
    dataname = [
        "musique",
        "2wikimqa",
        "multifieldqa_en",
        "narrativeqa",
        "longbook_qa_eng",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": -1,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop": None,
        "stop_token_ids": None,
        "ignore_eos": False,
        "max_tokens": args.max_output_len,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    cuda_num = 1
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=cuda_num,
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_input_len+2000,
    )

    for d in dataname:
        path = base_path + f"{d}.jsonl"

        with open(path, 'r') as file:
            data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
        for i in range(len(data)):
            if "id" not in data[i]:
                data[i]["id"] = (i+1)

        dataset = llmDataset(data, tokenizer, args.template, d, args.max_input_len)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.Collactor)

        # 输出路径
        output_path = args.output_dir + f"/{d}" + f"/preds_{d}_{args.template}.jsonl"

        # log路径
        log_path = args.output_dir + f"/{d}" + f"/logs_{d}_{args.template}.txt"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        log = open(log_path, 'w')
        log.write(f"Model loaded from {args.model_path}\n")

        cnt = 0
        total_subem_cot = 0
        total_subem_ans = 0
        total_em = 0
        total_f1 = 0
        total_len = 0
        max_len = 0

        with open(output_path, 'w') as outfile:
            for batch in tqdm(dataloader):
                input_prompt = batch['input_prompt']
                outputs: list = llm.generate(input_prompt, sampling_params)
                cleaned_outputs = [output.outputs[0].text for output in outputs]
                maxindex = len(batch['id'])
                for index in range(maxindex):
                    log.write(f"Sample {cnt + 1}\n")

                    id = batch['id'][index]
                    input = batch['input'][index]
                    prompt = input_prompt[index]
                    answers = batch['answers'][index]
                    pred = cleaned_outputs[index]

                    log.write(f"QA prompt: {prompt[:1000] + prompt[-1000:]}\n")
                    log.write(f"Ground truth answer: {answers}\n")
                    log.write(f"LLM QA Response: \n{pred}\n")

                    qa_em_score = 0.0
                    qa_f1_score = 0.0
                    qa_subem_cot_score = 0.0
                    qa_subem_ans_score = 0.0
                    if args.template == "direct_answer":
                        qa_em_score = drqa_exact_match_score(pred, answers)
                        qa_f1_score = F1_scorer(pred, answers)
                    elif args.template in ["cot_answer", "sealong"]:
                        pred_ans, flag = extract_answer(pred)
                        if flag:
                            qa_em_score = drqa_exact_match_score(pred_ans, answers)
                            qa_f1_score = F1_scorer(pred_ans, answers)
                            qa_subem_ans_score = substring_exact_match_score(pred_ans, answers)
                        else:
                            log.write("No answer found in the response\n")
                        qa_subem_cot_score = substring_exact_match_score(pred, answers)
                    else:
                        raise NotImplementedError("template error!")

                    log.write(f"QA EM score: {qa_em_score}\n")
                    log.write(f"QA F1 score: {qa_f1_score:.4f}\n")
                    log.write(f"QA SubEM CoT score: {qa_subem_cot_score}\n")
                    log.write(f"QA SubEM Ans score: {qa_subem_ans_score}\n")

                    total_subem_cot += qa_subem_cot_score
                    total_subem_ans += qa_subem_ans_score
                    total_em += qa_em_score
                    total_f1 += qa_f1_score
                    total_len += len(prompt)
                    max_len = max(max_len, len(prompt))

                    if (cnt + 1) % 50 == 0:
                        log.write("*" * 50 + "\n")
                        log.write(f"Prompt Length: {total_len / (cnt + 1):.4f}\n")
                        log.write(f"Max Prompt Length: {max_len}\n")
                        log.write(f"Total QA EM: {total_em / (cnt + 1):.4f}\n")
                        log.write(f"Total QA F1: {total_f1 / (cnt + 1):.4f}\n")
                        log.write(f"Total QA SubEM CoT: {total_subem_cot / (cnt + 1):.4f}\n")
                        log.write(f"Total QA SubEM Ans: {total_subem_ans / (cnt + 1):.4f}\n")
                    log.write("*" * 50 + "\n")

                    output_item = {
                        "id": id,
                        "input": input,
                        "answers": answers,
                        "pred": pred,
                    }

                    json_str = json.dumps(output_item, ensure_ascii=False)
                    outfile.write(json_str + '\n')
                    cnt += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="/data1/duanshaohua/hf_hub/models--meta-llama--Llama-3.1-8B-Instruct")
    parser.add_argument("--input_dir", type=str,
                        default="/data1/duanshaohua/hf_hub/datasets--THUDM--LongBench/processed/")
    parser.add_argument("--datasets", type=str, nargs='+',
                        default=None)
    parser.add_argument("--output_dir", type=str,
                        default="/data1/duanshaohua/rankcot/result/zero_shot/llama3.1/lb_v1")
    parser.add_argument("--template", type=str,
                        default="cot_answer", choices=["direct_answer", "cot_answer"])
    parser.add_argument("--max_input_len", type=int, default=100000)
    parser.add_argument("--max_output_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    inference(args)
