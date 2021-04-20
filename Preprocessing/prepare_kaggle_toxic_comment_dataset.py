# python file for creating custom dataset
# preparing kaggle toxic comment classification dataset
import csv
import config_with_yaml as config

cfg = config.load("config.yml")
directory = cfg.getProperty('Path.kaggle-toxic-comment-directory')


def prepare_training_file():
    fn = cfg.getProperty('Path.OutputFileName.train')
    training_path = directory + "train_prepro.csv"
    with open(training_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count, label_count = 0, [0, 0]
        dataset = []
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                label = 0
                for i in row[2:8]:
                    label = int(i) or label
                if label_count[label] > 15000:  # total 30000 rows with each label having 15000 each
                    continue
                label_count[label] += 1
                dataset.append((row[1], label))
                line_count += 1
            # if line_count > 5:
            #    break
        print(f'Processed {line_count} lines.')
        # print(dataset)
    with open(directory + fn, mode='w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['text', 'label'])
        for item in dataset:
            csv_writer.writerow(list(item))


def prepare_test_file():
    test_path = directory + "test_prepro.csv"
    test_labels = directory + "test_labels.csv"

    labels = {}
    with open(test_labels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                label = 0
                for i in row[1:7]:
                    label = int(i) or label
                labels[row[0]] = label
                line_count += 1
        #             if line_count > 5:
        #                break
        print(f'Processed {line_count} lines.')
        # print(labels)
    with open(test_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count, label_count = 0, [0, 0]
        dataset = []
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                label = labels.get(row[0])
                if label != -1:
                    if label_count[label] > 5000:  # total 10000 rows with each label having 5000 each
                        continue
                    label_count[label] += 1
                    dataset.append((row[1], labels[row[0]]))
                    line_count += 1
                else:
                    continue
        #             if line_count > 5:
        #                break
        print(f'Processed {line_count} lines.')
    #         print(dataset)
    fn = cfg.getProperty('Path.OutputFileName.test')
    with open(directory + fn, mode='w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['text', 'label'])
        for item in dataset:
            csv_writer.writerow(list(item))


prepare_training_file()
prepare_test_file()
