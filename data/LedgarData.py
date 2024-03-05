import sys

import pandas as pd

from data.DataClass import DataClass
from data.datafiles.ledgar_categories import LEDGAR_CATEGORIES
from eval.util import print_information


class LedgarData(DataClass):
    def __init__(self, data_source, prompts, tokenizer_name):
        super().__init__(data_source, prompts, data_config="ledgar", context_alias='<provision>',
                         tokenizer_name=tokenizer_name)
        self.ledgar_categories = LEDGAR_CATEGORIES
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        answers = []
        for provision, label in zip(df['text'], df['label']):
            if omit_ans:
                answers.append(self.ledgar_categories[label])
                answer = ''
            else:
                answer = self.ledgar_categories[label]
            full_input = self.generate_prompt(prompt=prompt, context=provision, options='',
                                              answer=answer)
            permutations.append(full_input)
        return permutations, answers

    def get_gold_standards(self):
        return self.test_input_df['label'].tolist()

    def resolve_label_numbers(self, model_predictions):
        self.test_input_df['model_predictions'] = model_predictions
        prediction_labels = []
        for provision, label, prediction in zip(self.test_input_df['text'], self.test_input_df['label'],
                                                self.test_input_df['model_predictions']):
            correct_answer = self.ledgar_categories[label].strip()
            if prediction == correct_answer:
                prediction_labels.append(label)
            else:
                prediction_labels.append(-1)
        return prediction_labels

    def evaluate_results(self, predictions):
        print(predictions)
        # df = pd.DataFrame({'predictions': predictions})
        # df.to_csv('ledgar_predictions.tsv', sep='\t', index=False)
        #
        # print(f'ledgar predictions size = {len(predictions)}')
        # prediction_labels = []
        # for prediction in predictions:
        #     answer = str(prediction).split('<assistant>:')[1].strip()
        #     prediction_labels.append(answer)
        # prediction_labels = self.resolve_label_numbers(prediction_labels)
        # gold_standards = self.get_gold_standards()
        # filename = 'ledgar_result.txt'
        # print_information(gold_standards, prediction_labels, filename)
