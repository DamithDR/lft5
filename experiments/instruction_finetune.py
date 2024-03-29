import argparse
import pickle

import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

import os

from config.lora_setting import CONFIG


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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
    file_path = args.file_path if args.file_path else 'data/permuted_data/'

    data_files = str(args.dataset_file_name).split(',')
    dataset = pd.DataFrame()
    if len(data_files) > 1:
        for data_file in data_files:
            d_set = pd.read_csv(f'{file_path}{data_file}', sep='\t')
            dataset = pd.concat([dataset, d_set], axis=0)
    else:
        dataset = pd.read_csv(f'{file_path}{args.dataset_file_name}', sep='\t')

    data = Dataset.from_pandas(dataset[['instructions']])
    splits = data.train_test_split(test_size=0.2, shuffle=True, seed=42)
    data = splits['train'].shuffle()
    validation_data = splits["test"].shuffle()

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        device_map="auto",
        # max_memory={0: "20GIB", 1: "20GIB"},
        # offload_folder="offload", offload_state_dict=True,
        trust_remote_code=True,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    tokenizer.pad_token = tokenizer.eos_token

    data = data.map(tokenize_prompt, batch_size=args.batch_size)
    validation_data = validation_data.map(tokenize_prompt, batch_size=args.batch_size)
    # if os.path.isfile('data.pkl'):
    #     with open('data.pkl', 'rb') as f:
    #         data = pickle.load(f)
    # else:
    #     data = data.map(tokenize_prompt, batch_size=args.batch_size)
    #     result_list = list(data)
    #
    #     with open('data.pkl', 'wb') as f:
    #         pickle.dump(result_list, f)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=CONFIG[args.model_type],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    training_args = transformers.TrainingArguments(
        gradient_accumulation_steps=4,
        # auto_find_batch_size=True,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=1000,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=False,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        save_total_limit=4,
        logging_steps=args.logging_steps,
        output_dir=args.model_output_path if args.model_output_path else "output_dir",
        save_strategy='steps',
        optim="adamw_torch",
        warmup_ratio=0.05,
        report_to="wandb"
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        eval_dataset=validation_data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.save_pretrained(f'finetuned_{str(args.model_name).replace("/", "_")}{args.model_output_path}')
    with open(f'finetuned_{str(args.model_name).replace("/", "_")}{args.model_output_path}/stat.txt', 'w') as f:
        f.write(f'Data files used : {",".join(data_files)}\n')
        f.write(f'model used : {args.model_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on legal instruction finetuning''')
    parser.add_argument('--model_type', type=str, required=True, help='model_type')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--dataset_file_name', type=str, required=True, help='comma separated dataset file names ')
    parser.add_argument('--file_path', type=str, required=False, help='directory path for the files',
                        default='data/permuted_data/')
    parser.add_argument('--cache_dir', type=str, required=False, help='cache directory')
    parser.add_argument('--model_output_path', type=str, required=False, help='model output directory',
                        default='output_dir')
    parser.add_argument('--eval_steps', type=int, default=64000, required=False, help='eval steps')
    parser.add_argument('--logging_steps', type=int, default=32000, required=False, help='logging steps')
    parser.add_argument('--batch_size', type=int, required=False, default=256,
                        help='training batch size')
    # parser.add_argument('--max_mem', type=str, required=True, help='max memory consumption per device')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.padding_side = "left"
    data_path = ''
    run(args)
