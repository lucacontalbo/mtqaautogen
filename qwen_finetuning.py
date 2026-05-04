import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Needed by PEFT + FSDP auto-wrap.
os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "Qwen3_5DecoderLayer"

import torch
import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


max_seq_length = 16384

OUTPUT_DIR = os.environ["OUTPUTDIR"]
MODELPATH = os.environ["MODELPATH"]

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
is_main_process = local_rank == 0

base_path = (
    "results/ours/Qwen3-5-122B-A10B-FP8/"
    "multi-table/{num_tables}/environmental/{method}/{perturbation}/data/reasoning.csv"
)

marker = "Let's think step-by-step."


# -------------------------
# Dataset construction
# -------------------------

data_samples = []

for num_tables in [2, 3, 5, 10, 20]:
    for method in ["average", "sum", "superlative"]:
        for perturbation in ["unit_converted", "not_unit_converted"]:
            path = base_path.format(
                num_tables=num_tables,
                method=method,
                perturbation=perturbation,
            )

            df = pd.read_csv(path)

            for _, row in df.iterrows():
                reasoning = row["CoT Reasoning"]

                try:
                    if marker not in reasoning:
                        raise ValueError("Missing step-by-step marker.")

                    if "</think>" not in reasoning:
                        raise ValueError("Missing </think> marker.")

                    input_text = reasoning.split(marker)[0] + marker

                    generated_text = marker.join(
                        reasoning.split(marker)[1:]
                    )

                    thinking_body = generated_text.split("</think>")[0]
                    answer_text = generated_text.split("</think>", 1)[1]

                    thinking_text = "<think>\n" + thinking_body + "</think>"

                    data_samples.append(
                        {
                            "prompt": input_text,
                            "completion": thinking_text + answer_text,
                        }
                    )

                except Exception:
                    if is_main_process:
                        print("Sample missing one of the required markers, skipping.")


if is_main_process:
    print(f"Loaded {len(data_samples)} samples")

train_samples, valid_samples = train_test_split(
    data_samples,
    test_size=0.05,
    random_state=0,
    shuffle=True,
)

if is_main_process:
    print(f"Train samples: {len(train_samples)}, Validation samples: {len(valid_samples)}")

dataset = DatasetDict(
    {
        "train": Dataset.from_list(train_samples),
        "validation": Dataset.from_list(valid_samples),
    }
)


# -------------------------
# Tokenizer
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODELPATH,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


def keep_example(example):
    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        example["prompt"] + example["completion"],
        add_special_tokens=False,
    )["input_ids"]

    return len(prompt_ids) < max_seq_length and len(full_ids) > len(prompt_ids)


dataset = dataset.filter(
    keep_example,
    num_proc=1,
)

if is_main_process:
    print(dataset)


# -------------------------
# Model
# -------------------------

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODELPATH,
    dtype=compute_dtype,
    attn_implementation="sdpa",
)

model.config.use_cache = False


decoder_layer_classes = sorted(
    {
        module.__class__.__name__
        for module in model.modules()
        if "DecoderLayer" in module.__class__.__name__
    }
)

if is_main_process:
    print("Detected decoder layer classes:", decoder_layer_classes)

if "Qwen3_5DecoderLayer" not in decoder_layer_classes:
    raise ValueError(
        f"Expected Qwen3_5DecoderLayer, but found {decoder_layer_classes}. "
        "Update FSDP_TRANSFORMER_CLS_TO_WRAP and transformer_layer_cls_to_wrap."
    )


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

model = get_peft_model(model, peft_config)

# Do not call model.gradient_checkpointing_enable() with FSDP
# when fsdp_config["activation_checkpointing"] is True.

if is_main_process:
    model.print_trainable_parameters()


# -------------------------
# Trainer
# -------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    args=SFTConfig(
        max_length=max_seq_length,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        warmup_steps=10,
        num_train_epochs=1,
        logging_steps=10,

        output_dir=OUTPUT_DIR,
        optim="adamw_torch",

        seed=0,
        dataset_num_proc=1,

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        completion_only_loss=True,
        packing=False,
        report_to="none",

        # Must be False when using FSDP activation_checkpointing.
        gradient_checkpointing=False,

        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": "Qwen3_5DecoderLayer",
            "use_orig_params": True,
            "activation_checkpointing": True,
            "limit_all_gathers": True,
        },

        save_strategy="epoch",
        eval_strategy="no",
    ),
)


trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
