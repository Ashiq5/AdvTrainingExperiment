from datasets import load_dataset
import csv
import config_with_yaml as config
cfg = config.load("../config.yml")


def prepare_adversarial_texts(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        adv_texts, ground_truth_labels = [], []
        for row in csv_reader:
            if line_count == 0:
                continue
            else:
                adv_texts.append(row[0])
                ground_truth_labels.append(row[1])
                line_count += 1
    return adv_texts, ground_truth_labels


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


def return_dataset(name):
    dataset = load_dataset_from_file('csv', name)

    train_val_dataset = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_val_dataset['train']
    validation_dataset = train_val_dataset['test']

    test_dataset = dataset['test']

    return train_dataset, validation_dataset, test_dataset
