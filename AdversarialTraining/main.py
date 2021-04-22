from args import Args
from prepare_dataset import return_dataset, prepare_dataset_for_training, prepare_adversarial_texts
from model import lstm_model, cnn_model
from prepare_dataloader import _make_dataloader
from training import train
from evaluate import evaluate
from attack import attack
from generate_adversarial_samples import _generate_adversarial_examples
import torch
import csv
import datetime
from textattack.models.wrappers import PyTorchModelWrapper


def train_evaluate_attack(model_wrapper, adversarial_training=True, model_name_prefix=None):
    if adversarial_training:
        # training the model
        trained_model, train_losses = train(args, model_wrapper, data_loaders=[adv_train_dataloader, eval_dataloader],
                                            pre_dataset=(
                                                train_text + adv_train_text, train_labels + ground_truth_labels))
    else:
        # training the model
        trained_model, train_losses = train(args, model_wrapper, data_loaders=[train_dataloader, eval_dataloader],
                                            pre_dataset=(train_text, train_labels))

    # saving the model
    output_dir = "models/"
    model_name = model_name_prefix + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_path = output_dir + model_name + ".pt"
    torch.save(trained_model.state_dict(), model_path)
    model_wrapper = PyTorchModelWrapper(trained_model, tokenizer)

    # reloading it from disk for evaluation
    model.load_state_dict(torch.load(model_path))
    test_accuracy = evaluate(model, test_dataloader)

    # now, test the success rate of attack_class_for_testing on this adv. trained model
    performance = attack(model_wrapper, args, list(zip(test_text, test_labels)))
    return train_losses, test_accuracy, performance


def save_samples_in_csv(fn):
    with open('adv_samples/' + fn, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Adversarial Text', 'Ground Truth Output'])

        for idx, text in enumerate(adv_train_text):
            csv_writer.writerow([text, ground_truth_labels[idx]])


def save_result_in_csv(fn, details):
    with open('result/' + fn, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Original Text', 'Perturbed Text', 'Attack Output',
                             'Original Output', 'Ground Truth Output', 'Result'])

        for detail in details:
            csv_writer.writerow(detail)


def get_args_for_model():
    # create args
    # You just need to change the parameters here
    attack_classes = ["TextFoolerJin2019", "BAEGarg2019", "TextBuggerLi2018"]
    at = False
    if at:
        return Args(dataset="kaggle-toxic-comment", batch_size=32, epochs=75,
                    model_short_name="lstm", num_attack_samples=500,
                    adversarial_samples_to_train=2000,
                    orig_model_prefix="lstm-kaggle-toxic-comment-", attack_class_for_training=attack_classes[2],
                    adv_sample_file="lstm-kaggle-textbugger.csv", adversarial_training=False, model_path=None)
    else:
        return Args(attack_class_for_training=attack_classes[2], attack_class_for_testing=attack_classes[2],
                    dataset="kaggle-toxic-comment", batch_size=32, epochs=75,
                    num_attack_samples=500,
                    model_short_name="lstm", at_model_prefix="lstm-at-textbugger-kaggle-toxic-comment-",
                    adv_sample_file="lstm-kaggle-bae.csv", adversarial_training=True)


def load_model_from_disk(arg):
    model.load_state_dict(torch.load(arg.model_path))
    model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    print(evaluate(model_wrapper.model, test_dataloader))  # for checking whether loading is correct
    return model_wrapper


if __name__ == "__main__":
    load_from_disk = True  # change this
    args = get_args_for_model()

    # define model and tokenizer
    if args.model_short_name == "lstm":
        model_wrapper = lstm_model(args)
    else:
        model_wrapper = cnn_model(args)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # prepare dataset and dataloader
    train_dataset, validation_dataset, test_dataset = return_dataset()
    train_text, train_labels = prepare_dataset_for_training(train_dataset)
    eval_text, eval_labels = prepare_dataset_for_training(validation_dataset)
    test_text, test_labels = prepare_dataset_for_training(test_dataset)

    train_dataloader = _make_dataloader(
        tokenizer, train_text, train_labels, args.batch_size
    )
    eval_dataloader = _make_dataloader(
        tokenizer, eval_text, eval_labels, args.batch_size
    )
    test_dataloader = _make_dataloader(
        tokenizer, test_text, test_labels, args.batch_size
    )

    if args.adversarial_training:
        adv_train_text, ground_truth_labels = prepare_adversarial_texts("adv_samples/" + args.adv_sample_file)
        adv_train_dataloader = _make_dataloader(
            tokenizer, train_text + adv_train_text, train_labels + ground_truth_labels, args.batch_size
        )

    # now train
    if not load_from_disk:
        train_losses, test_accuracy, performance = train_evaluate_attack(model_wrapper,
                                                                         model_name_prefix=args.orig_model_prefix,
                                                                         adversarial_training=args.adversarial_training)
        if args.adversarial_training:
            prefix = args.at_model_prefix
        else:
            prefix = args.orig_model_prefix

        with open('result/' + prefix + '-loss.txt', 'w') as f:
            for idx, loss in enumerate(train_losses):
                f.write("%d %lf\n" % (idx, loss))

        with open('result/' + prefix + '-performance.txt', 'w') as f:
            f.write("Test Accuracy: %lf\n" % test_accuracy)
            for item in performance[0]:
                f.write(item[0] + " " + str(item[1]) + "\n")

        save_result_in_csv(args.orig_model_prefix + '-details.csv', performance[1])

    # pre-generate and save adversarial samples in a csv
    model_wrapper = load_model_from_disk(args)
    adv_train_text, ground_truth_labels = _generate_adversarial_examples(model_wrapper,
                                                                         args,
                                                                         list(zip(train_text, train_labels)),
                                                                         save=args.adv_sample_file)
    # save_samples_in_csv(args.adv_sample_file)
