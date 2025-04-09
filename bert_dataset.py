import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

def load_and_prepare_dataset(file_path, remove_none_targets=False):
    df = pd.read_csv(file_path)

    df = df[['text', 'label', 'target']]

    if remove_none_targets:
        # Only remove "none" and "notgiven" if doing bias classification
        df = df[(df['target'] != 'none') & (df['target'] != 'notgiven')]

    # Map labels: hate -> 1, nothate -> 0
    df['label'] = df['label'].apply(lambda x: 1 if x == 'hate' else 0)

    # Map targets
    target_labels = df['target'].unique().tolist()
    target2id = {label: i for i, label in enumerate(target_labels)}
    id2target = {i: label for label, i in target2id.items()}
    df['target_id'] = df['target'].map(target2id)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(tokenize_function, batched=True)

    return dataset, target_labels, target2id, id2target
