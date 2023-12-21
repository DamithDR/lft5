import json

from datasets import load_dataset

from data.DataClass import DataClass


class MultiEURLEXData(DataClass):

    def __init__(self, data_source, prompts, data_config='en'):
        super().__init__(data_source=data_source, data_config=data_config, prompts=prompts, context_alias='<law>')
        self.dataset = load_dataset(self.data_source, self.data_config, split='train')
        self.classlabel = self.dataset.features["labels"].feature
        with open('data/datafiles/eurovoc_descriptors.json') as jsonl_file:
            self.eurovoc_concepts = json.load(jsonl_file)
        self.filter_dataset()

    def get_labels(self, labels):
        return [self.eurovoc_concepts[self.classlabel.int2str(int(label))][self.data_config] for label in labels]

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for law, labels in zip(df['text'], df['labels']):
            label_lst = self.get_labels(list(labels))
            if omit_ans:
                answer = ''
            else:
                answer = ",".join(label_lst)

            full_input = self.generate_prompt(prompt=prompt, context=law, options='',
                                              answer=answer)
            permutations.append(full_input)
        return permutations

