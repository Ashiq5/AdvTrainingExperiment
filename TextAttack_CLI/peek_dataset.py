# peek-dataset command, write the file path in config.yml and write down the file name in config.yml
import csv
import config_with_yaml as config
cfg = config.load("config.yml")


def peek_dataset(fn):
    path = cfg.getProperty('Path.kaggle-toxic-comment-directory')
    path = path + fn
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        dataset = []
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                dataset.append((row[0], int(row[1])))
                line_count += 1
            # if line_count > 5:
            #    break
        print(f'Processed {line_count} lines.')
        # print(dataset)
        return dataset


dataset = peek_dataset(cfg.getProperty('peek-which'))
