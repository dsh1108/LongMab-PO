# LongMab-PO
Long-Context LLM Preference Optimization
Source code for paper: Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization
## Overview
LongMab-PO is a novel framework that leverages a Multi-Armed Bandit (MAB) rollout strategy to identify the most informative chunks from the given long context for sampling high-quality and diverse responses and constructing preference data pairs for Direct Preference Optimization (DPO) training.

![](fig/LongMab_final.png)

## Requirements

### 1. Requirement.
**Install the following packages using Pip or Conda under this environment**

```
Python==3.10.14
torch==2.2.1
transformers==4.40.0
faiss==1.8.0
tqdm
trl==0.8.6
vllm==0.4.1
accelerate==0.30.1
deepspeed==0.14.2
peft==0.10.0
```

### 2. Install LLaMA-Factory.
Refer to [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for detailed instructions.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## LongMab-PO Pipeline

### 1. Prepare the Training Data
You can follow [SeaLong](https://github.com/SihengLi99/SEALONG/tree/main) to synthesize your own original training data, or download the file from [here](https://drive.google.com/drive/folders/1QJ63-90RIdjyKwAdCMZKLz5KiFfxEkoq?usp=sharing) and place them in the `data/` directory.
 **Each sample must contain the following four required fields:**

```json
{
  "id": "A unique identifier for the sample (int)",
  "question": "The input question (str)",
  "answer": "The ground truth answer to the question (str)",
  "context": "The synthesized long context (str)"
}     
```


### 2. Run the LongMab-PO Pipeline
**(1) Generate Probe CoT:**
```
conda activate llama3_inf
python src/answer_generation/querypassage_to_CoT.py \
--model_path  # The path to RankCoT model \
--data_path # e.g. nq_modify10passage \
--output_name # e.g. nq_querypassage_to_CoT.jsonl
--max_psg_length 1500
```
**(2) Running the Multi-Armed Bandit Rollout Process:**
**(3) Construct Preference Data Pairs:**
**(4) Train the Model:**
### 3. Evaluation
cd 
