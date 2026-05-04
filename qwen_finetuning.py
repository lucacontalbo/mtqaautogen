from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import pandas as pd

max_seq_length = 16384
OUTPUT_DIR = "finetuned_models/qwen35_9b_reasoning_finetuned_lora"

base_path = "results/ours/Qwen3-5-122B-A10B-FP8/multi-table/{num_tables}/environmental/{method}/{perturbation}/data/reasoning.csv"
data_samples = []
for num_tables in [2, 3, 5, 10, 20]:
    for method in ["average", "sum", "superlative"]:
        for perturbation in ["unit_converted", "not_unit_converted"]:
            path = base_path.format(num_tables=num_tables, method=method, perturbation=perturbation)
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                data_sample = {}
                reasoning = row["CoT Reasoning"]
                try:
                    input_text = reasoning.split("Let's think step-by-step.")[0]+"Let's think step-by-step."
                    generated_text = "Let's think step-by-step.".join(reasoning.split("Let's think step-by-step.")[1:])
                    thinking_text = "<think>\n"+generated_text.split("</think>")[0]+"</think>"
                    answer_text = generated_text.split("</think>")[1]
                    data_sample["messages"] = [
                        {
                            "role": "user",
                            "content": input_text
                        },
                        {
                            "role": "assistant",
                            "content": thinking_text + answer_text
                        }
                    ]
                    data_samples.append(data_sample)
                except:
                    print("Sample missing one of the required markers, skipping.")

print(f"Loaded {len(data_samples)} samples")
train_samples, valid_samples = train_test_split(
    data_samples,
    test_size=0.05,
    random_state=0,
    shuffle=True,
)

print(f"Train samples: {len(train_samples)}, Validation samples: {len(valid_samples)}")

dataset = DatasetDict({
    "train": Dataset.from_list(train_samples),
    "validation": Dataset.from_list(valid_samples),
})

"""dataset = load_dataset(
    "json",
    data_files={"train": "train_reasoning.jsonl", "validation": "valid_reasoning.jsonl"},
)"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3.5-9B", # TODO: change model name
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    load_in_16bit = True,
    full_finetuning = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0.0,
    bias = "none",
    # "unsloth" checkpointing is intended for very long context + lower VRAM
    use_gradient_checkpointing = "unsloth",
    random_state = 0,
    max_seq_length = max_seq_length,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    processing_class = tokenizer,
    args = SFTConfig(
        max_seq_length = max_seq_length, # or max_length
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs=1,
        logging_steps = 10,
        output_dir = OUTPUT_DIR,
        optim = "adamw_8bit",
        seed = 0,
        dataset_num_proc = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),

        assistant_only_loss = True,
        packing = False,
        report_to = "none",
    ),
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)