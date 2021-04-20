from args import Args
from prepare_dataset import return_dataset, prepare_dataset_for_training
from model import lstm_model
from prepare_dataloader import _make_dataloader
from training import train
from evaluate import evaluate
from attack import attack
import torch
import datetime
from textattack.models.wrappers import PyTorchModelWrapper


if __name__ == "__main__":
    # create args
    attack_classes = ["TextFoolerJin2019", "BAEGarg2019", "TextBuggerLi2018"]
    args = Args(attack_class_for_training=attack_classes[1], attack_class_for_testing=attack_classes[0],
                dataset="kaggle-toxic-comment", batch_size=32, epochs=100,
                adversarial_samples_to_train=500, attack_period=50)

    # prepare dataset
    train_dataset, validation_dataset, test_dataset = return_dataset()
    train_text, train_labels = prepare_dataset_for_training(train_dataset)
    eval_text, eval_labels = prepare_dataset_for_training(validation_dataset)
    test_text, test_labels = prepare_dataset_for_training(test_dataset)

    # define model and tokenizer
    model_wrapper = lstm_model(args)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

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

    # training the model
    trained_model = train(args, model_wrapper, data_loaders=[train_dataloader, eval_dataloader, test_dataloader],
                          pre_dataset=(train_text, train_labels))

    # saving the model
    output_dir = "models/"
    model_name = "lstm-at-bae-kaggle-toxic-comment-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_path = output_dir + model_name + ".pt"
    torch.save(trained_model.state_dict(), output_dir + model_name + ".pt")
    model_wrapper = PyTorchModelWrapper(trained_model, tokenizer)

    # reloading it from disk for evaluation
    model.load_state_dict(torch.load(model_path))
    evaluate(model, test_dataloader)

    # now, test the success rate of attack_class_for_testing on this adv. trained model
    attack(model_wrapper, args, list(zip(test_text, test_labels)))