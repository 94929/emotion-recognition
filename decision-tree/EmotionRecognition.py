import numpy as np
import pickle
import scipy.io as sio
import sys

from IE3 import choose_best_decision_attribute
from Stats import build_confusion_matrix, combine_confusion_matrices, get_stats_from_confusion_matrix, \
    visualise_confusion_matrix
from TreeVisualizer import visualize_ete_tree


# from TreeVisualizer import visualize_ete_tree


class DecisionTree:
    def __init__(self, v, o, k):
        self.value = v
        self.op = o
        self.kids = k

    def classify(self, muscles):
        if self.value is not None:
            return self.value
        else:
            return self.kids[muscles[self.op]].classify(muscles)


# return mode of binary targets, is that just a regular mode?
def majority_value_mode(binary_targets):
    return max(set(binary_targets), key=binary_targets.count)


# return mean of binary targets
def majority_value_average(binary_targets):
    return sum(binary_targets) / len(binary_targets) if len(binary_targets) > 0 else 0


# returns true if all are the same value, fail fast algorithm
def same_value(binary_targets):
    first_target = binary_targets[0]
    for target in binary_targets:
        if target != first_target:
            return False
    return True


# learns and returns a decision tree based on the examples, binary_targets and majority_function provided
def decision_tree_learning(examples, attributes, binary_targets, majority_function):
    if len(binary_targets) == 0 or len(examples) == 0:
        # should only happen if initially passed an empty argument
        return DecisionTree(0, None, [])
    elif same_value(binary_targets):
        # return a leaf node with value = binary_targets[0]
        return DecisionTree(binary_targets[0], None, [])
    elif attributes is None or len(attributes) == 0:
        # return a leaf node with value = majority_value(binary_targets)
        return DecisionTree(majority_function(binary_targets), None, [])
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        tree = DecisionTree(None, best_attribute, [])
        attributes.remove(best_attribute)
        for i in range(2):
            new_examples = []
            new_targets = []
            for j in range(len(examples)):
                if examples[j][best_attribute] == i:
                    new_examples.append(examples[j])
                    new_targets.append(binary_targets[j])
            if len(new_examples) == 0:
                return DecisionTree(majority_function(binary_targets), None, [])
            else:
                tree.kids.append(decision_tree_learning(new_examples, attributes, new_targets, majority_function))
        return tree


# builds a tree for each emotion and returns all of them in a list
def emotion_trees(examples, emotions, majority_function):
    trees = []
    for emotion in range(1, 7):
        binary_targets = list(map(lambda target: 1 if target == emotion else 0, emotions))
        trees.append(decision_tree_learning(examples, list(range(45)), binary_targets, majority_function))
    return trees


def classify_with_average(trees, a, b):
    correct = 0
    for i in range(len(a)):
        highest_probability_emotion = (-1, -1, None)
        for index, tree in enumerate(trees):
            current_probability_emotion = (tree.classify(a[i]), index + 1)
            highest_probability_emotion = max(highest_probability_emotion, current_probability_emotion)
        predicted_label = highest_probability_emotion[1]
        if predicted_label == b[i]:
            correct += 1

    print("Classified correctly " + str(correct) + " out of " + str(len(a)))


# returns a list of predictions based on the classification of the 6 trees
def test_trees(trees, features):
    predictions = []
    for feature in features:
        tree_outputs = [tree.classify(feature) for tree in trees]
        predicted_emotion = 0
        best_prediction = 0
        for i in range(len(tree_outputs)):
            prediction = tree_outputs[i]
            if prediction > best_prediction:
                best_prediction = prediction
                predicted_emotion = i
        predictions.append(predicted_emotion + 1)
    return predictions


def calculate_classifier_error(predicted_labels, correct_labels):
    error_sum = 0
    for i in range(len(predicted_labels)):
        correct_label = correct_labels[i]
        predicted_label = predicted_labels[i]
        if correct_label != predicted_label:
            error_sum += 1
    return (1.0 / len(predicted_labels)) * error_sum


# calculate total error estimate using k-fold cross validation
def cross_validator(data, results, k, majority_function):
    segment_size = int(len(data) / k)
    data_segments = [data[i:i + segment_size] for i in range(0, len(data) - (len(data) % segment_size), segment_size)]
    results_segments = [results[i:i + segment_size] for i in
                        range(0, len(results) - (len(results) % segment_size), segment_size)]

    total_confusion_matrix = None
    total_classification_rate = 0
    for i in range(k):
        test_data = data_segments[i]
        test_results = results_segments[i]
        training_data = []
        training_results = []
        for j in range(len(data_segments)):
            if i != j:
                for l in range(len(data_segments[j])):
                    training_data.append(data_segments[j][l])
                    training_results.append(results_segments[j][l])

        training_data = np.array(training_data)
        training_results = np.array(training_results)

        trees = emotion_trees(training_data, training_results, majority_function)
        tree_predictions = test_trees(trees, test_data)

        confusion_matrix = build_confusion_matrix(tree_predictions, test_results)
        # stats = get_stats_from_confusion_matrix(confusion_matrix)

        if total_confusion_matrix is None:
            total_confusion_matrix = confusion_matrix
        else:
            total_confusion_matrix = combine_confusion_matrices(total_confusion_matrix, confusion_matrix)

        error = calculate_classifier_error(tree_predictions, test_results)
        # print("Error is " + str(error))
        classification_rate = 1.0 - error
        # print("Classification rate is " + str(classification_rate))
        total_classification_rate += classification_rate
    avg_stats = get_stats_from_confusion_matrix(total_confusion_matrix)
    print("Confusion matrix: ")
    visualise_confusion_matrix(total_confusion_matrix)
    print("Recall, Precision, F1 Measure for emotions 1 to 6")
    for avg_stat in avg_stats:
        print(avg_stat)
    avg_classification_rate = total_classification_rate / k
    print()
    print("Average classification rate is " + str(avg_classification_rate))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Pass path to data file as command argument e.g. python3 EmotionRecognition.py "
              "Data/cleandata_students.mat")
    else:
        mat = sio.loadmat(sys.argv[1])
        x = mat['x']
        y = list(map(lambda f: f[0], mat['y']))
        trees_from_data = emotion_trees(x, y, majority_value_average)
        print("Printing generated trees for each emotion")
        for t in range(len(trees_from_data)):
            print("For emotion {}".format(t + 1))
            visualize_ete_tree(trees_from_data[t])
        print("Dumping each tree to pickle files")
        for t in range(len(trees_from_data)):
            file_object = open("tree{}.pkl".format(t + 1), 'wb')
            pickle.dump(trees_from_data[t], file_object)
            file_object.close()

        print("Performing cross-validation using majority_value_mode function")
        cross_validator(x, y, 10, majority_value_mode)

        print()

        print("Performing cross-validation using majority_value_average function")
        cross_validator(x, y, 10, majority_value_average)

        print()
        print("Using the inputted data set and the majority_value_average:")
        classify_with_average(trees_from_data, x, y)
