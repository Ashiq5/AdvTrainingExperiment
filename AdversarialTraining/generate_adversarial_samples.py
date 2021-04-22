from textattack.attack_results import SuccessfulAttackResult
import textattack
from tqdm import tqdm
from textattack.attack_recipes import TextFoolerJin2019, BAEGarg2019, TextBuggerLi2018
import csv


def _generate_adversarial_examples(model, args, dataset, save=False):
    """Create a dataset of adversarial examples based on perturbations of the
    existing dataset.

    :param model: Model to attack.
    :param attack_class: class name of attack recipe to run.
    :param dataset: iterable of (text, label) pairs.

    :return: list(AttackResult) of adversarial examples.
    """
    attack = eval(args.attack_class_for_training).build(model)
    adv_train_text, ground_truth_labels = [], []
    num_successes = 0
    if isinstance(save, str):
        with open('adv_samples/' + save, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Adversarial Text', 'Ground Truth Output'])
    for idx, result in enumerate(attack.attack_dataset(dataset)):
        print(idx, " no. attack tried")
        if isinstance(result, SuccessfulAttackResult):
            adv_train_text.append(result.perturbed_text())
            ground_truth_labels.append(result.original_result.ground_truth_output)
            csv_writer.writerow([result.perturbed_text(), result.original_result.ground_truth_output])
            num_successes += 1
            print(num_successes, " attack succeeded")
        if num_successes >= args.adversarial_samples_to_train:
            break

    return adv_train_text, ground_truth_labels
