from args import Args
from prepare_dataset import return_dataset, prepare_dataset_for_training
from model import lstm_model, cnn_model
from prepare_dataloader import _make_dataloader
from training import train
from evaluate import evaluate
from attack import attack
from generate_adversarial_samples import _generate_adversarial_examples
import torch
import datetime
from textattack.models.wrappers import PyTorchModelWrapper

import config_with_yaml as config
cfg = config.load("../config.yml")


def train_evaluate_attack(model_wrapper, adversarial_training=True, model_name_prefix=None):
    if not adversarial_training:
        args.attack_class_for_training = None

    # training the model
    trained_model = train(args, model_wrapper, data_loaders=[train_dataloader, eval_dataloader],
                          pre_dataset=(train_text, train_labels))

    # saving the model
    output_dir = "models/"
    model_name = model_name_prefix + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_path = output_dir + model_name + ".pt"
    torch.save(trained_model.state_dict(), model_path)
    model_wrapper = PyTorchModelWrapper(trained_model, tokenizer)

    # reloading it from disk for evaluation
    model.load_state_dict(torch.load(model_path))
    evaluate(model, test_dataloader)

    # now, test the success rate of attack_class_for_testing on this adv. trained model
    performance = attack(model_wrapper, args, list(zip(test_text, test_labels)))
    return performance


def save_in_csv(fn, adv_text, labels):
    import csv
    path = cfg.getProperty('Path.adv_sample_path')
    with open('dataset/' + fn, mode='w') as csv_file:
        employee_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(['Adversarial Text', 'Ground Truth Output'])

        for idx, text in enumerate(adv_text):
            employee_writer.writerow([adv_text[idx], labels[idx]])


if __name__ == "__main__":
    # create args
    attack_classes = ["TextFoolerJin2019", "BAEGarg2019", "TextBuggerLi2018"]
    # You just need to change the parameters here
    args = Args(attack_class_for_training=attack_classes[1], attack_class_for_testing=attack_classes[0],
                dataset="kaggle-toxic-comment", batch_size=32, epochs=100,
                adversarial_samples_to_train=2000, attack_period=50, num_attack_samples=50,
                model_short_name="lstm", at_model_prefix="lstm-at-bae-kaggle-toxic-comment-",
                orig_model_prefix="lstm-kaggle-toxic-comment-")

    # prepare dataset
    train_dataset, validation_dataset, test_dataset = return_dataset()
    train_text, train_labels = prepare_dataset_for_training(train_dataset)
    eval_text, eval_labels = prepare_dataset_for_training(validation_dataset)
    test_text, test_labels = prepare_dataset_for_training(test_dataset)

    # define model and tokenizer
    if args.model_short_name == "lstm":
        model_wrapper = lstm_model(args)
    else:
        model_wrapper = cnn_model(args)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # pre-generate and save adversarial samples in a csv
    adv_train_text, ground_truth_labels = _generate_adversarial_examples(model_wrapper,
                                                                         args,
                                                                         list(zip(train_text, train_labels)))
    save_in_csv("lstm-kaggle-bae.csv", adv_train_text, ground_truth_labels)  # change this name b4 running
    exit()

    # prepare dataloader
    train_dataloader = _make_dataloader(
        tokenizer, train_text, train_labels, args.batch_size
    )
    eval_dataloader = _make_dataloader(
        tokenizer, eval_text, eval_labels, args.batch_size
    )
    test_dataloader = _make_dataloader(
        tokenizer, test_text, test_labels, args.batch_size
    )

    # adversarial
    at_performance = train_evaluate_attack(model_wrapper, model_name_prefix=args.at_model_prefix)

    # now do non-adversarially to compare with the old attack performance
    non_at_performance = train_evaluate_attack(model_wrapper, model_name_prefix=args.orig_model_prefix,
                                               adversarial_training=False)

    print(at_performance)
    print(non_at_performance)
