import argparse

import pandas as pd
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def run(args):
    config = PeftConfig.from_pretrained("finetuned_tiiuae_falcon-7b-instruct/")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path)

    model_inf = PeftModel.from_pretrained(model, "finetuned_tiiuae_falcon-7b-instruct")

    # set teh generation configuration params
    gen_config = model_inf.generation_config
    gen_config.max_new_tokens = 200
    gen_config.temperature = 0.2
    gen_config.top_p = 0.7
    gen_config.num_return_sequences = 1
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id

    dataset = pd.read_csv(f'data/permuted_data/{args.dataset_file_name}', sep='\t')
    dataset = dataset[300:400]

    out_list = []
    for data in dataset['instructions']:
        prompt = data.split('<assistant>')[0]

        # encode the prompt
        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        # do the inference
        with torch.inference_mode():
            outputs = model.generate(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask,
                                     generation_config=gen_config)
        out_list.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    predictions = pd.DataFrame({'gold': dataset['instructions'], 'predictions': out_list})
    predictions.to_csv('predictions.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models arabic readability assessment''')
    parser.add_argument('--dataset_file_name', type=str, required=True, help='dataset_name')
    args = parser.parse_args()
    run(args)
