import json
import sys

import pandas as pd
from datasets import load_dataset

from data.DataClass import DataClass
from eval.utilities.retrieval_metrics import mean_rprecision


class MultiEURLEXData(DataClass):

    def __init__(self, data_source, prompts, tokenizer_name):
        super().__init__(data_source=data_source, data_config='en', prompts=prompts, context_alias='<law>',
                         tokenizer_name=tokenizer_name)
        self.dataset = load_dataset(self.data_source, self.data_config, split='train')
        self.classlabel = self.dataset.features["labels"].feature
        with open('data/datafiles/eurovoc_descriptors.json') as jsonl_file:
            self.eurovoc_concepts = json.load(jsonl_file)
        self.filter_dataset()

    def get_labels(self, labels):
        return [self.eurovoc_concepts[self.classlabel.int2str(int(label))][self.data_config] for label in labels]

    def get_ints_from_strings(self, predictions):
        prediction_labels = []
        for label in predictions:
            try:
                value = self.classlabel.str2int(int(label))
            except ValueError:
                value = -1
            prediction_labels.append(value)
        return prediction_labels

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        answers = []
        for law, labels in zip(df['text'], df['labels']):
            label_lst = self.get_labels(list(labels))
            if omit_ans:
                answers.append(",".join(label_lst))
                answer = ''
            else:
                answer = ",".join(label_lst)

            full_input = self.generate_prompt(prompt=prompt, context=law, options='',
                                              answer=answer)
            permutations.append(full_input)
        return permutations, answers

    def get_gold_standards(self):
        return self.test_input_df['labels'].tolist()

    def resolve_labels(self, prediction_labels):
        resolved_labels = []

        for law, correct_answers, preds in zip(self.test_input_df['text'], self.test_input_df['labels'],
                                               prediction_labels):

            label_list = self.get_ints_from_strings(preds)

            if len(label_list) < len(correct_answers):
                label_list.extend([-1] * (len(correct_answers) - len(label_list)))
            elif len(label_list) > len(correct_answers):
                label_list = label_list[:len(correct_answers)]
            resolved_labels.append(label_list)
        return resolved_labels

    def evaluate_results(self, predictions):
        print(predictions)
        # df = pd.DataFrame({'predictions': predictions})
        # df.to_csv('multieurlex_predictions.tsv', sep='\t', index=False)
        #
        # print(f'multieurlex predictions size = {len(predictions)}')
        # prediction_labels = []
        # for prediction in predictions:
        #     answer = str(prediction).split('<assistant>:')[1].strip()
        #     labels = answer.split(',')
        #     labels = [label.strip() for label in labels]
        #     prediction_labels.append(labels)
        # y_pred = self.resolve_labels(prediction_labels)
        # y_true = self.get_gold_standards()
        # scores = f' R-Precision: {mean_rprecision(y_true, y_pred)[0] * 100:2.2f}\t'
        # print(scores)
        # filename = 'multieurlex_result.txt'
        # with open(filename,'w') as f:
        #     f.write(scores)
        # sys.exit(0)
