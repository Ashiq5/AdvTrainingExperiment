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


def just_train(model_wrapper, adversarial_training=True, model_name_prefix=None):
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
    # just train the model and save it
    """
    model_wrapper = PyTorchModelWrapper(trained_model, tokenizer)

    # reloading it from disk for evaluation
    model.load_state_dict(torch.load(model_path))
    test_accuracy = evaluate(model, test_dataloader)

    # now, test the success rate of attack_class_for_testing on this adv. trained model
    performance = attack(model_wrapper, args, list(zip(test_text, test_labels)))
    """
    return train_losses  # , test_accuracy, performance


def just_evaluate(model_wrapper):
    test_accuracy = evaluate(model_wrapper.model, test_dataloader)
    # now, test the success rate of attack_class_for_testing on this adv. trained model
    performance = attack(model_wrapper, args, list(zip(test_text, test_labels)))
    return test_accuracy, performance


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


def get_args():
    # create args
    # You just need to change the parameters here
    attack_classes = ["TextFoolerJin2019", "BAEGarg2019", "TextBuggerLi2018"]
    at = False  # Todo: change here
    if not at:  # normal training
        return Args(dataset="imdb", model_short_name="lstm",
                    batch_size=32, epochs=75,
                    adversarial_training=False,
                    orig_model_prefix="lstm-imdb-",
                    max_length=2500,

                    # evaluate
                    attack_class_for_testing=attack_classes[1],  # test robustness against which model
                    num_attack_samples=50,  # how many samples to test robustness with

                    # pre-generate
                    attack_class_for_training=attack_classes[1],
                    # launch which attack to generate adv samples on the trained model
                    adv_sample_file="lstm-kaggle-bae.csv",  # file name of where to save adv. samples
                    adversarial_samples_to_train=2000,  # how many samples in adv_sample_file
                    )
    else:  # adversarial training
        return Args(dataset="kaggle-toxic-comment", model_short_name="lstm",
                    batch_size=32, epochs=30,
                    adversarial_training=True,
                    at_model_prefix="lstm-at-tb-kaggle-toxic-comment-",
                    adv_sample_file="lstm-kaggle-textbugger.csv",

                    # evaluate
                    attack_class_for_testing=attack_classes[1],
                    num_attack_samples=500,
                    )


def load_model_from_disk():
    model.load_state_dict(torch.load("/home/grads/iashiq5/AdvTrainingExperiment/AdversarialTraining/models/lstm-kaggle-toxic-comment-2021-04-22-15-46-30-739480.pt"))  # Todo: provide file name here
    model_wrapper = PyTorchModelWrapper(model, tokenizer)
    print(evaluate(model_wrapper.model, test_dataloader))  # for checking whether loading is correct
    return model_wrapper


if __name__ == "__main__":
    # 3 tasks: train, evaluate, pre-generate
    task = "train"  # Todo: change this
    args = get_args()

    # define model and tokenizer
    if args.model_short_name == "lstm":
        model_wrapper = lstm_model(args)
    else:
        model_wrapper = cnn_model(args)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # prepare dataset and dataloader
    train_dataset, validation_dataset, test_dataset = return_dataset(args.dataset)
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

    # either load_from_disk or train
    if task == "evaluate":
        model_wrapper = load_model_from_disk()
        test_accuracy, performance = just_evaluate(model_wrapper)

        if args.adversarial_training:
            prefix = args.at_model_prefix
        else:
            prefix = args.orig_model_prefix

        prefix += args.attack_class_for_testing

        with open('result/' + prefix + '-performance.txt', 'w') as f:
            f.write("Test Accuracy: %lf\n" % test_accuracy)
            for item in performance[0]:
                f.write(item[0] + " " + str(item[1]) + "\n")

        save_result_in_csv(prefix + '-details.csv', performance[1])

    elif task == "train":
        if args.adversarial_training:
            prefix = args.at_model_prefix
        else:
            prefix = args.orig_model_prefix
        train_losses = just_train(model_wrapper,
                                  model_name_prefix=prefix,
                                  adversarial_training=args.adversarial_training)

        with open('result/' + prefix + '-loss.txt', 'w') as f:
            for idx, loss in enumerate(train_losses):
                f.write("%d %lf\n" % (idx, loss))

    elif task == "pre-generate":  # pre-generate and save adversarial samples in a csv
        model_wrapper = load_model_from_disk()
        adv_train_text, ground_truth_labels = _generate_adversarial_examples(model_wrapper,
                                                                             args,
                                                                             list(zip(train_text, train_labels)),
                                                                             save=args.adv_sample_file)
    # save_samples_in_csv(args.adv_sample_file)
