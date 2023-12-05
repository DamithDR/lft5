import json

import pandas as pd
from datasets import load_dataset

from data.DataClass import DataClass


class MultiEURLEXData(DataClass):

    def __init__(self, data_source, prompts, data_config='en'):
        super().__init__(data_source=data_source, data_config=data_config, prompts=prompts, context_alias='<law>')
        self.dataset = load_dataset(self.data_source, self.data_config, split='train')
        self.classlabel = self.dataset.features["labels"].feature
        with open('data/datafiles/eurovoc_descriptors.json') as jsonl_file:
            self.eurovoc_concepts = json.load(jsonl_file)

    def get_labels(self, labels):
        return [self.eurovoc_concepts[self.classlabel.int2str(int(label))][self.data_config] for label in labels]

    def generate_permutations(self):
        permutations = []

        for prompt in self.prompts:
            for law, labels in zip(self.input_df['text'], self.input_df['labels']):
                label_lst = self.get_labels(list(labels))

                full_input = self.generate_prompt(prompt=prompt, context=law, options='',
                                                  answer=",".join(label_lst))
                permutations.append(full_input)

        return pd.DataFrame({'instructions': permutations})
