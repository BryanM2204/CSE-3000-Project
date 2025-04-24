import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

def load_and_prepare_dataset(file_path, binary_bias=False):
    df = pd.read_csv(file_path)
    df = df[['text', 'label', 'target']]

    if binary_bias:
        # Binary bias model: 1 = group targeted, 0 = none/notgiven
        df['bias'] = df['target'].apply(lambda t: 0 if t == 'none' else 1)
    else:
        df = df[(df['target'] != 'none') & (df['target'] != 'notgiven')]
        df['label'] = df['label'].apply(lambda x: 1 if x == 'hate' else 0)
        target_labels = df['target'].unique().tolist()
        target2id = {label: i for i, label in enumerate(target_labels)}
        df['target_id'] = df['target'].map(target2id)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(tokenize_function, batched=True)

    if binary_bias:
        dataset['train'] = dataset['train'].rename_column("bias", "labels")
        dataset['test'] = dataset['test'].rename_column("bias", "labels")
        return dataset
    else:
        return dataset, target_labels, target2id, None
