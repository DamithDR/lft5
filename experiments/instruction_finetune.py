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
    tok_full_prompt = tokenizer(text_input['instructions'], padding=True, truncation=False)
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
    data_files = str(args.dataset_file_name).split(',')
    dataset = pd.DataFrame()
    if len(data_files) > 1:
        for data_file in data_files:
            d_set = pd.read_csv(f'data/permuted_data/{data_file}', sep='\t')
            dataset = pd.concat([dataset, d_set], axis=0)
    else:
        dataset = pd.read_csv(f'data/permuted_data/{args.dataset_file_name}', sep='\t')
    data = Dataset.from_pandas(dataset[['instructions']])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: args.max_mem, 1: args.max_mem, 2: args.max_mem, 3: args.max_mem, "cpu": "120GiB"},
        offload_folder="offload", offload_state_dict=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    data = data.map(tokenize_prompt)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

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
        report_to="wandb"
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.save_pretrained(f'finetuned_{str(args.model_name).replace("/", "_")}_{",".join(data_files)}')
    with open(f'finetuned_{str(args.model_name).replace("/", "_")}_{",".join(data_files)}/stat.txt', 'w') as f:
        f.write(f'Data files used : {",".join(data_files)}\n')
        f.write(f'model used : {args.model_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on legal instruction finetuning''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset_file_name', type=str, required=True, help='comma separated dataset file names ')
    parser.add_argument('--max_mem', type=str, required=True, help='max memory consumption per device')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_path = ''
    run(args)
