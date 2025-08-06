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

## Training LongMab-PO

### 1. Prepare the Original Training Data
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
cd scripts
bash gen_probe.sh
```
**(2) Running the Multi-Armed Bandit Rollout Process:**
```
bash rollout.sh
```
**(3) Construct Preference Data Pairs:**
```
bash filter.sh
```

### 3. Train the Model
You can train the model by utilizing LLaMA-Factory framework quickly, we provide the yaml files. Please refer to LLaMA-Factory for relevant environment installation and configuration.
```
cd Llama-Factory
```
You can also download the checkpoint of [Llama-3.1-8B-Instruct](https://huggingface.co/sanwuge/ConsJudge-qwen) and [Qwen-2.5-7B-Instruct](https://huggingface.co/sanwuge/ConsJudge-llama) directly.

## 3. Evaluation
cd 
