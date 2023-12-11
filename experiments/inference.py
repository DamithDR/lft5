import argparse

import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    model_inf = PeftModel.from_pretrained(model, args.model_name)

    # set teh generation configuration params
    gen_config = model_inf.generation_config
    gen_config.max_new_tokens = 200
    gen_config.temperature = 0.2
    gen_config.top_p = 0.7
    gen_config.num_return_sequences = 1
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id

    dataset = pd.read_csv(f'data/permuted_data/{args.test_file_name}', sep='\t')
    dataset = dataset[:20]
    out_list = []

    num = 0
    data_list = dataset['instructions'].to_list()
    total_no = len(dataset)
    for i in range(0, total_no, args.batch_size):
        prev_num = num
        num = num + args.batch_size
        data_batch = data_list[prev_num:num]
        print(f'processing : {num}/{total_no}')
        # encode the prompt
        encoding = tokenizer(data_batch, padding=True, truncation=False, return_tensors="pt").to(model.device)
        # do the inference
        with torch.inference_mode():
            outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
                                     generation_config=gen_config)
            print(outputs)
        out_list.extend(tokenizer.decode(outputs, skip_special_tokens=True))

    predictions = pd.DataFrame({'gold': dataset['instructions'], 'predictions': out_list})
    flat_model_name = str(args.model_name).replace('/', '')
    predictions.to_csv(f'{flat_model_name}_{args.test_file_name}_predictions.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models on legal instruction finetuning''')
    parser.add_argument('--test_file_name', type=str, required=True, help='dataset_name')
    parser.add_argument('--model_name', type=str, required=True, help='model_name_or_path')
    parser.add_argument('--batch_size', type=int, default=10, required=False, help='inference batch size')

    args = parser.parse_args()
    run(args)
