import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import set_seed

# Assuming you have a custom dataset with train and test splits
train_dataset = pd.read_csv('data/datafiles/hub/policy_train_data.csv', sep='\t')
test_dataset = pd.read_csv('data/datafiles/hub/policy_test_data.csv', sep='\t')
test_dataset = test_dataset.drop(['Ann1', 'Ann2', 'Ann3', 'Ann4', 'Ann5', 'Ann6'], axis=1)
test_dataset = test_dataset.rename(columns={'Any_Relevant': 'Label'})
custom_dataset = DatasetDict({'train': Dataset.from_pandas(train_dataset), 'test': Dataset.from_pandas(test_dataset)})

# Set the Hugging Face Hub username and organization (replace with your own values)
hf_username = 'Damith'
hf_organization = 'LegalLLMs'

# Set the dataset name
dataset_name = 'LegalLLMs/privacy-qa'

# Push the dataset to the Hugging Face Hub
custom_dataset.push_to_hub(
    repo_id=dataset_name,
    private=True,
    commit_message="data upload",
    token='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'  # Replace with your Hugging Face API token
)
