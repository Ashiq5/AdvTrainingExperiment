import torch
from training import device


def _get_test_score(dataloader, model, loss_function):
    correct_predictions, total_predictions, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # Step 1. Get inputs and labels for a particular batch and load them to device
            input_ids, labels = batch
            labels = labels.to(device)
            input_ids = input_ids.to(device)

            # Step 2: Predict/ Forward pass
            logits = model(input_ids)

            # Step 3: Calculate prediction in test set and add to total loss
            loss = loss_function(logits, labels)
            pred_labels = logits.argmax(dim=-1)
            correct_predictions += (pred_labels == labels).sum().item()
            total_predictions += len(pred_labels)
            total_loss += loss.item()

    return correct_predictions / total_predictions, total_loss / len(dataloader.dataset)


def evaluate(model, test_dataloader):
    test_accuracy, test_loss = _get_test_score(test_dataloader, model, torch.nn.CrossEntropyLoss())
    print("Test Accuracy: ", test_accuracy, "Test Loss: ", test_loss)
