import argparse
import gc

import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def run(args):
    config = PeftConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_inf = PeftModel.from_pretrained(model, args.model_name)

    # set teh generation configuration params
    gen_config = model_inf.generation_config
    gen_config.max_new_tokens = 200
    gen_config.temperature = 0.2
    gen_config.top_p = 0.7
    gen_config.num_return_sequences = 1
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id

    # dataset = pd.read_csv(f'data/permuted_data/{args.test_file_name}', sep='\t')

    data_files = [
        'Test_casehold.tsv',
        'Test_ledgar.tsv',
        'Test_multi_eurlex.tsv',
        'Test_multi_lexsum.tsv',
        'Test_privacy_qa.tsv',
        'Test_scotus.tsv',
        'Test_uk_abs_in_abs.tsv'
    ]
    dataset = pd.DataFrame()
    if len(data_files) > 1:
        dataset_name = 'all'
        for data_file in data_files:
            d_set = pd.read_csv(f'data/permuted_data/{data_file}', sep='\t')
            dataset = pd.concat([dataset, d_set], axis=0)
    else:
        dataset_name = str(args.test_file_name).split('.')[0]
        dataset = pd.read_csv(f'data/permuted_data/{args.test_file_name}', sep='\t')

    out_list = []

    num = 0
    data_list = dataset['instructions'].to_list()
    total_no = len(dataset)
    with torch.inference_mode():
        for i in range(0, total_no, args.batch_size):
            prev_num = num
            num = num + args.batch_size
            data_batch = data_list[prev_num:num]
            print(f'processing : {num}/{total_no}')
            # encode the prompt
            encoding = tokenizer(data_batch, padding=True, truncation=False, return_tensors="pt").to(model.device)
            # do the inference
            outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
                                     generation_config=gen_config)
            detach = outputs.detach().cpu().numpy()
            outputs = detach.tolist()
            out_list.extend([tokenizer.decode(out, skip_special_tokens=True) for out in outputs])
            clear_gpu_memory()

    predictions = pd.DataFrame({'gold': dataset['instructions'], 'predictions': out_list})
    flat_model_name = str(args.model_name).replace('/', '')
    predictions.to_csv(f'{flat_model_name}_{dataset_name}_predictions.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on legal instruction finetuning''')
    parser.add_argument('--test_file_name', type=str, required=False, help='dataset_name')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--batch_size', type=int, default=10, required=False, help='inference batch size')

    args = parser.parse_args()
    run(args)
