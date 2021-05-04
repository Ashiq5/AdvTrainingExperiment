import csv

with open("details/cnn-kaggle-toxic-comment-TextFoolerJin2019--details.csv") as file:
    csv_reader = csv.reader(file, delimiter=',')
    total, success, failure, skipped, orig, at = 0, 0, 0, 0, 0, 0
    for row in csv_reader:
        # if row[4] != '1':
        #     continue
        if row[5] == 'Succeeded':
            success += 1
        elif row[5] == 'Failed':
            failure += 1
        else:
            skipped += 1
        if row[2] == row[4]:
            at += 1
        if row[3] == row[4]:
            orig += 1
        total += 1
    print(success, failure, skipped)
    print("Success Rate:", success/(success+failure) * 100)
    print("Original Accuracy", (total-skipped)/total * 100)
    print("After Attack Accuracy:", failure/total * 100)
