import pandas as pd

from data.DataClass import DataClass
from data.datafiles.scotus_categories import categories


class ScotusData(DataClass):

    def __init__(self, data_source, prompts, tokenizer_name):
        super().__init__(data_source, prompts, data_config="scotus", context_alias='{opinion}',
                         tokenizer_name=tokenizer_name)
        self.scotus_categories = categories
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        answers = []
        for opinion, label in zip(df['text'], df['label']):
            if omit_ans:
                answers.append(self.scotus_categories[label])
                answer = ''
            else:
                answer = self.scotus_categories[label]
            full_input = self.generate_prompt(prompt=prompt, context=opinion, options='',
                                              answer=answer)
            permutations.append(full_input)
        return permutations, answers

    def get_gold_standards(self):
        return self.test_input_df['label'].tolist()

    def evaluate_results(self, predictions):
        df = pd.DataFrame({'predictions': predictions})
        df.to_csv('scouts_predictions.tsv', sep='\t', index=False)

        print(f'scouts predictions size = {len(predictions)}')
