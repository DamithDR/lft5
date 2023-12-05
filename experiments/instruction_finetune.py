import argparse

import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training


def tokenize_inputs(text_input):
    tok_full_prompt = tokenizer(text_input, padding=True, truncation=False)
    return tok_full_prompt


def tokenize_prompt(text_input):
    tok_full_prompt = tokenizer(text_input, padding=True, truncation=False)
    return tok_full_prompt


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def run(args):
    dataset = pd.read_csv(f'data/permuted_data/{args.dataset_file_name}')
    data = Dataset.from_pandas(dataset)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # quantization_config=bnb_config,
        #     load_in_8bit=True,  #if you want to load the 8-bit model
        #     device_map='auto',
        device_map="auto",
        offload_folder="offload", offload_state_dict=True,
        # max_memory={0: args.max_mem_consumption},  # , 2: "20GIB", 3: "20GIB"},#, "cpu": "60GiB"},
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    data = data.map(tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    training_args = transformers.TrainingArguments(
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=4,
        logging_steps=25,
        output_dir="output_dir",  # give the location where you want to store checkpoints
        save_strategy='epoch',
        optim="paged_adamw_8bit",
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models arabic readability assessment''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset_file_name', type=str, required=True, help='dataset_name')
    parser.add_argument('--max_mem_consumption', type=str, required=True, help='max memory consumption per device')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_path = ''
    run(args)