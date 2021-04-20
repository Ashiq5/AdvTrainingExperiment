import torch
from AdversarialTraining import main
from generate_adversarial_samples import _generate_adversarial_examples
from prepare_dataloader import _make_dataloader

device = main.device

def need_grad(x):
    return x.requires_grad


def epoch(dataloader, model, loss_function, optimizer):
    correct_predictions, total_predictions, total_loss = 0, 0, 0.0
    for step, batch in enumerate(dataloader):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get inputs and labels for a particular batch and load them to device
        input_ids, labels = batch
        labels = labels.to(device)
        input_ids = input_ids.to(device)

        # Step 3: Predict/ Forward pass
        logits = model(input_ids)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        # Step 5: Calculate prediction in training set add to total loss
        pred_labels = logits.argmax(dim=-1)
        correct_predictions += (pred_labels == labels).sum().item()
        total_predictions += len(pred_labels)
        total_loss += loss.item()

    return correct_predictions / total_predictions, total_loss / len(dataloader.dataset)


def train(args, model_wrapper, data_loaders=None, pre_dataset=None):
    optimizer = torch.optim.Adam(filter(need_grad, model_wrapper.model.parameters()), lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    epochs = args.epochs
    num_gpus = torch.cuda.device_count()
    train_dataloader = data_loaders[0]
    print("Num GPUs", num_gpus)

    for epoch_no in range(0, epochs):
        train_acc, train_loss = epoch(train_dataloader, model_wrapper.model, loss_function, optimizer)
        print("Epoch:", epoch, "Loss: ", train_loss, "Train Acc: ", train_acc * 100, "%")

        if epoch_no % args.attack_period == 0:
            print("Generating adversarial samples at epoch ", epoch)
            adv_train_text, ground_truth_labels = _generate_adversarial_examples(model_wrapper,
                                                                                 args,
                                                                                 list(zip(pre_dataset[0], pre_dataset[1])))
            print("Adding adversarial samples to the training data")
            train_dataloader = _make_dataloader(
                model_wrapper.tokenizer, adv_train_text + pre_dataset[0], ground_truth_labels + pre_dataset[1],
                args.batch_size
            )


