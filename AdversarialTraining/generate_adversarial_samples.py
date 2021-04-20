from textattack.attack_results import SuccessfulAttackResult
import textattack
from tqdm import tqdm
from textattack.attack_recipes import TextFoolerJin2019, BAEGarg2019, TextBuggerLi2018


def _generate_adversarial_examples(model, args, dataset):
    """Create a dataset of adversarial examples based on perturbations of the
    existing dataset.

    :param model: Model to attack.
    :param attack_class: class name of attack recipe to run.
    :param dataset: iterable of (text, label) pairs.

    :return: list(AttackResult) of adversarial examples.
    """
    attack = eval(args.attack_class_for_training).build(model)
    adv_train_text, ground_truth_labels = [], []
    results_iterable = attack.attack_dataset(dataset)
    num_successes = 0

    for num_successes in tqdm(range(args.adversarial_samples_to_train)):
        try:
            result = next(results_iterable)
            if isinstance(result, SuccessfulAttackResult):
                adv_train_text.append(result.perturbed_text())
                ground_truth_labels.append(result.original_result.ground_truth_output)
                num_successes += 1
        except StopIteration:
            break

    return adv_train_text, ground_truth_labels
