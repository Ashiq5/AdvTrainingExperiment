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


def train_evaluate_attack(model_wrapper, adversarial_training=True, model_name_prefix=None, method="pre-generate"):
    if not adversarial_training:
        args.attack_class_for_training = None
    if method == "pre-generate":
        args.attack_class_for_training = None
        # training the model
        trained_model, train_losses = train(args, model_wrapper, data_loaders=[adv_train_dataloader, eval_dataloader],
                                            pre_dataset=(train_text + adv_train_text, train_labels + ground_truth_labels))
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
    evaluate(model, test_dataloader)

    # now, test the success rate of attack_class_for_testing on this adv. trained model
    performance = attack(model_wrapper, args, list(zip(test_text, test_labels)))
    return train_losses, performance


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


if __name__ == "__main__":
    # create args
    attack_classes = ["TextFoolerJin2019", "BAEGarg2019", "TextBuggerLi2018"]
    # You just need to change the parameters here
    args = Args(attack_class_for_training=attack_classes[1], attack_class_for_testing=attack_classes[0],
                dataset="kaggle-toxic-comment", batch_size=32, epochs=100,
                adversarial_samples_to_train=3, attack_period=50, num_attack_samples=500,
                model_short_name="lstm", at_model_prefix="lstm-at-bae-kaggle-toxic-comment-",
                orig_model_prefix="lstm-kaggle-toxic-comment-",
                adv_sample_file="lstm-kaggle-bae.csv")

    # prepare dataset
    train_dataset, validation_dataset, test_dataset = return_dataset()
    train_text, train_labels = prepare_dataset_for_training(train_dataset)
    eval_text, eval_labels = prepare_dataset_for_training(validation_dataset)
    test_text, test_labels = prepare_dataset_for_training(test_dataset)
    # adv_train_text, ground_truth_labels = prepare_adversarial_texts("adv_samples/" + args.adv_sample_file)

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
    save_samples_in_csv(args.adv_sample_file)
    exit()

    # prepare dataloader
    if args.adversarial_training:
        adv_train_dataloader = _make_dataloader(
            tokenizer, train_text + adv_train_text, train_labels + ground_truth_labels, args.batch_size
        )
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
    at_train_losses, at_performance = train_evaluate_attack(model_wrapper, model_name_prefix=args.at_model_prefix)

    # define model and tokenizer again for training non-adversarially, gets retrained otherwise
    if args.model_short_name == "lstm":
        model_wrapper = lstm_model(args)
    else:
        model_wrapper = cnn_model(args)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    # now do non-adversarially to compare with the old attack performance
    orig_train_losses, orig_performance = train_evaluate_attack(model_wrapper,
                                                                model_name_prefix=args.orig_model_prefix,
                                                                adversarial_training=False)

    with open('result/' + args.at_model_prefix + '-loss.txt', 'w') as f:
        for idx, loss in enumerate(at_train_losses):
            f.write("%lf %lf\n" % (idx, loss))

    with open('result/' + args.orig_model_prefix + '-loss.txt', 'w') as f:
        for idx, loss in enumerate(orig_train_losses):
            f.write("%lf %lf\n" % (idx, loss))

    with open('result/' + args.at_model_prefix + '-performance.txt', 'w') as f:
        for item in at_performance[0]:
            f.write("%s %lf\n" % (item[0], item[1]))

    with open('result/' + args.orig_model_prefix + '-performance.txt', 'w') as f:
        for item in orig_performance[0]:
            f.write("%s %lf\n" % (item[0], item[1]))

    save_result_in_csv(args.at_model_prefix + '-details.csv', at_performance[1])
    save_result_in_csv(args.orig_model_prefix + '-details.csv', orig_performance[1])

