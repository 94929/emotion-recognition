import math


def entropy(positive_examples, negative_examples):
    p = len(positive_examples)
    n = len(negative_examples)

    if p + n == 0:
        return 0

    frac_p = p / (p + n)
    frac_n = n / (p + n)
    entropy_p = - (frac_p * math.log(frac_p, 2)) if p > 0 else 0
    entropy_n = - (frac_n * math.log(frac_n, 2)) if n > 0 else 0
    return entropy_p + entropy_n


def separate_positive_negative_examples(examples, binary_targets):
    positive_examples_indexes = []
    negative_examples_indexes = []
    for i in range(len(examples)):
        if binary_targets[i]:
            positive_examples_indexes.append(i)
        else:
            negative_examples_indexes.append(i)
    return positive_examples_indexes, negative_examples_indexes


# Calculate remainder for given attribute
def calculate_remainder(examples, positive_examples_indexes, negative_examples_indexes, attribute):
    p_0 = [index for index in positive_examples_indexes if not examples[index][attribute]]
    p_1 = [index for index in positive_examples_indexes if examples[index][attribute]]
    n_0 = [index for index in negative_examples_indexes if not examples[index][attribute]]
    n_1 = [index for index in negative_examples_indexes if examples[index][attribute]]

    remainder = ((len(p_0) + len(n_0)) / (len(positive_examples_indexes) + len(negative_examples_indexes))) * \
                entropy(p_0, n_0) + ((len(p_1) + len(n_1)) / (len(positive_examples_indexes) +
                                                              len(negative_examples_indexes))) * entropy(p_1, n_1)
    return remainder


# chooses the attribute that results in the highest information gain.
def choose_best_decision_attribute(examples, attributes, binary_targets):
    # Separate examples into positive and negative
    positive_examples_indexes, negative_examples_indexes = separate_positive_negative_examples(examples, binary_targets)
    i_p_n = entropy(positive_examples_indexes, negative_examples_indexes)

    max_gain = 0
    best_attribute = -1

    for attribute in attributes:
        gain = i_p_n - calculate_remainder(examples, positive_examples_indexes, negative_examples_indexes, attribute)
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute

    return best_attribute if max_gain > 0 else attributes[0]
