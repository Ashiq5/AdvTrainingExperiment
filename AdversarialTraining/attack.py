import textattack
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
from textattack.attack_recipes import TextFoolerJin2019, BAEGarg2019, TextBuggerLi2018
import numpy as np


def log_summary(results):
    total_attacks = len(results)
    if total_attacks == 0:
        return
    # Count things about attacks.
    all_num_words = np.zeros(len(results))
    perturbed_word_percentages = np.zeros(len(results))
    num_words_changed_until_success = np.zeros(
        2 ** 16
    )  # @ TODO: be smarter about this
    failed_attacks = 0
    skipped_attacks = 0
    successful_attacks = 0
    max_words_changed = 0
    details = []
    flag = ""
    for i, result in enumerate(results):
        all_num_words[i] = len(result.original_result.attacked_text.words)
        if isinstance(result, FailedAttackResult):
            failed_attacks += 1
            flag = "Failed"
            details.append((result.original_text(), result.perturbed_text(), result.perturbed_result.output,
                            result.original_result.output, result.original_result.ground_truth_output, flag))
            continue
        elif isinstance(result, SkippedAttackResult):
            skipped_attacks += 1
            flag = "Skipped"
            details.append((result.original_text(), result.perturbed_text(), result.perturbed_result.output,
                            result.original_result.output, result.original_result.ground_truth_output, flag))
            continue
        else:
            successful_attacks += 1
            flag = "Succeeded"
            details.append((result.original_text(), result.perturbed_text(), result.perturbed_result.output,
                            result.original_result.output, result.original_result.ground_truth_output, flag))
        num_words_changed = len(
            result.original_result.attacked_text.all_words_diff(
                result.perturbed_result.attacked_text
            )
        )
        num_words_changed_until_success[num_words_changed - 1] += 1
        max_words_changed = max(
            max_words_changed or num_words_changed, num_words_changed
        )
        if len(result.original_result.attacked_text.words) > 0:
            perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
            )
        else:
            perturbed_word_percentage = 0
        perturbed_word_percentages[i] = perturbed_word_percentage

    # Original classifier success rate on these samples.
    original_accuracy = (total_attacks - skipped_attacks) * 100.0 / total_attacks
    original_accuracy = str(round(original_accuracy, 2)) + "%"

    # New classifier success rate on these samples.
    accuracy_under_attack = failed_attacks * 100.0 / (total_attacks)
    accuracy_under_attack = str(round(accuracy_under_attack, 2)) + "%"

    # Attack success rate.
    if successful_attacks + failed_attacks == 0:
        attack_success_rate = 0
    else:
        attack_success_rate = (
                successful_attacks * 100.0 / (successful_attacks + failed_attacks)
        )
    attack_success_rate = str(round(attack_success_rate, 2)) + "%"

    perturbed_word_percentages = perturbed_word_percentages[
        perturbed_word_percentages > 0
        ]
    average_perc_words_perturbed = perturbed_word_percentages.mean()
    average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"

    average_num_words = all_num_words.mean()
    average_num_words = str(round(average_num_words, 2))

    summary_table_rows = [
        ["Number of successful attacks:", str(successful_attacks)],
        ["Number of failed attacks:", str(failed_attacks)],
        ["Number of skipped attacks:", str(skipped_attacks)],
        ["Original accuracy:", original_accuracy],
        ["Accuracy under attack:", accuracy_under_attack],
        ["Attack success rate:", attack_success_rate],
        ["Average perturbed word %:", average_perc_words_perturbed],
        ["Average num. words per input:", average_num_words],
    ]

    num_queries = np.array(
        [
            r.num_queries
            for r in results
            if not isinstance(r, SkippedAttackResult)
        ]
    )
    avg_num_queries = num_queries.mean()
    avg_num_queries = str(round(avg_num_queries, 2))
    summary_table_rows.append(["Avg num queries:", avg_num_queries])

    return summary_table_rows, details


def attack(model, args, dataset):
    """Create a dataset of adversarial examples based on perturbations of the
    existing dataset.

    :param model: Model to attack.
    :param attack_class: class name of attack recipe to run.
    :param dataset: iterable of (text, label) pairs.

    :return: list(AttackResult) of adversarial examples.
    """
    attack_model = eval(args.attack_class_for_testing).build(model)
    results_iterable = attack_model.attack_dataset(dataset)
    results = []
    num_attacks = 0
    for result in results_iterable:
        print(num_attacks, " attack tried")
        if num_attacks >= args.num_attack_samples:
            break
        results.append(result)
        num_attacks += 1
    return log_summary(results)
