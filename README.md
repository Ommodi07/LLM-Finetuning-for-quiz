---
base_model: mistralai/Mistral-7B-v0.1
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:mistralai/Mistral-7B-v0.1
- lora
- sft
- transformers
- trl
---

# Model Card for Model ID

## Model Details
Quiz Generation task based model

### Model Description

This model is a fine-tuned version of Mistral-7B-v0.1 specifically designed to generate multiple-choice questions (MCQs) for Machine Learning and Deep Learning topics. The model uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning with 4-bit quantization, making it memory-efficient while maintaining generation quality. It can generate questions with various difficulty levels, provide multiple options, identify correct answers, and include explanations.

- **Developed by:** [Ommodi07](https://github.com/Ommodi07)
- **Model type:** Causal Language Model with LoRA adapters
- **Language(s) (NLP):** English
- **Finetuned from model:** mistralai/Mistral-7B-v0.1

## Model download link (if unable to download from github)
**[Download](https://drive.google.com/drive/folders/1fgOTbf3--NLjB_MpzGg3JPcPBzDmPktQ?usp=sharing)**

## Uses

### Direct Use

This model can be directly used to generate multiple-choice questions for educational purposes, specifically for Machine Learning and Deep Learning topics. Users can specify the subject, topic, and difficulty level to get tailored quiz questions with options, correct answers, and explanations.

### Downstream Use

This model can be integrated into:
- Educational platforms for automatic quiz generation
- Learning management systems (LMS)
- Study aid applications
- Assessment tools for ML/DL courses

## Bias, Risks and Limitations

### Limitation : 
- Subject specific only
- Generate 1 question per prompt

## How to Get Started with the Model

Use the code below to get started with the model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Setup 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "quiz-lora-adapter")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("quiz-lora-adapter")

# Generate a quiz question
prompt = """### Instruction:
Generate a multiple-choice question.

Subject: Deep Learning
Topic: Convolutional Neural Networks
Difficulty: Medium

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Details

### Training Data

The model was trained on a custom dataset containing multiple-choice questions about Machine Learning and Deep Learning. Each training example includes:
- Subject and topic classification
- Difficulty level (Easy, Medium, Hard)
- Question text
- Multiple choice options
- Correct answer index
- Detailed explanation

Example : 
```
[
  {
    "id": "supervised_learning_mcq_211",
    "subject": "Machine Learning",
    "topic": "Supervised Learning",
    "type": "mcq",
    "question": "In the context of supervised learning, what is the purpose of a grid search?",
    "options": [
      "To search for the best combination of hyperparameters for a given model",
      "To split the data into training and testing sets",
      "To select the best features for a model",
      "To combine multiple models into an ensemble"
    ],
    "correct_option_index": 0,
    "explanation": "The purpose of a grid search in supervised learning is to search for the best combination of hyperparameters for a given model. Hyperparameters are the parameters of a model that are not learned from the data, but instead must be set by the user. Grid search involves systematically testing different combinations of hyperparameters to find the combination that results in the best performance.",
    "source": "aionlinecourse",
    "difficulty": "medium"
  },
]
```

Data was formatted using an instruction-following template to teach the model to generate structured quiz questions.

#### Training Hyperparameters

- **Training regime:** 4-bit quantization with NF4
- **Per device train batch size:** 1
- **Gradient accumulation steps:** 8
- **Learning rate:** 2e-4
- **Number of epochs:** 3
- **Optimizer:** paged_adamw_8bit
- **LR scheduler:** cosine
- **Warmup ratio:** 0.05
- **Max sequence length:** 1024
- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

<!-- #### Speeds, Sizes, Times [optional]

## Evaluation


### Testing Data, Factors & Metrics

#### Testing Data


#### Factors


#### Metrics

### Results


#### Summary

## Model Examination [optional]
 -->
## Technical Specifications

### Model Architecture and Objective

The model uses Mistral-7B-v0.1 as the base architecture with LoRA (Low-Rank Adaptation) adapters for parameter-efficient fine-tuning. The objective is causal language modeling, specifically trained to generate structured multiple-choice questions in an instruction-following format.

Key architectural features:
- 4-bit quantization using bitsandbytes (NF4 quantization type)
- LoRA adapters with rank 16 applied to attention and MLP layers
- Gradient checkpointing enabled for memory efficiency

### Compute Infrastructure

Colab : T4 GPU 

#### Hardware

Requires CUDA-enabled GPU for inference and training. The 4-bit quantization makes it possible to run on consumer GPUs with limited VRAM.

#### Software

- transformers
- accelerate
- peft
- bitsandbytes
- datasets
- trl (Transformer Reinforcement Learning)
- torch
