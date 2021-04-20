from datasets import load_dataset
import config_with_yaml as config

cfg = config.load("config.yml")


def prepare_dataset_for_training(datasets_dataset):
    """Changes an `datasets` dataset into the proper format for
    tokenization."""
    texts = [x['text'] for x in datasets_dataset]
    outputs = [x['label'] for x in datasets_dataset]
    return texts, outputs


def load_dataset_from_file(type_of_file, dataset_name):
    files_path = cfg.getProperty('Path.' + dataset_name)
    data_files = {}
    for path in files_path.keys():
        data_files[path] = files_path[path]
    dataset = load_dataset(type_of_file, data_files=data_files)
    return dataset


def return_dataset():
    dataset = load_dataset_from_file('csv', 'kaggle-toxic-comment')

    train_val_dataset = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_val_dataset['train']
    validation_dataset = train_val_dataset['test']

    test_dataset = dataset['test']

    return train_dataset, validation_dataset, test_dataset
