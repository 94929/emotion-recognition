import unittest
from EmotionRecognition import majority_value_mode
from EmotionRecognition import same_value
from EmotionRecognition import DecisionTree
from EmotionRecognition import decision_tree_learning

from IE3 import calculate_remainder
from IE3 import choose_best_decision_attribute
from IE3 import entropy
from IE3 import separate_positive_negative_examples


class IE3TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.examples = [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        self.binary_targets = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        self.attributes = [0, 1, 2, 3]

    def test_entropy_function(self):
        """Is entropy calcuated correctly?"""
        p = [0] * 5
        n = [1] * 9
        computed_entropy = entropy(p, n)
        self.assertAlmostEqual(0.94, computed_entropy, 2)

    def test_entropy_function_with_empty_arguments(self):
        """Is the entropy of two empty lists calculated properly?"""
        self.assertEqual(entropy([], []), 0)

    def test_correctly_separates_positive_negative_examples(self):
        """Are examples separated into positive and negative correctly?"""
        p, n = separate_positive_negative_examples(self.examples, self.binary_targets)
        self.assertEqual(len(p), 9)
        self.assertEqual(len(n), 5)

    def test_correctly_calculates_remainder(self):
        """Is the remainder calculated correctly?"""
        p, n = separate_positive_negative_examples(self.examples, self.binary_targets)
        remainder = calculate_remainder(self.examples, p, n, 0)
        self.assertAlmostEqual(0.838, remainder, 2)

    def test_choose_best_attribute_function(self):
        """Is the best attribute chosen correctly?"""
        self.assertEqual(choose_best_decision_attribute(self.examples, self.attributes, self.binary_targets), 2)


class EmotionRecognitionTestCase(unittest.TestCase):

    def assertTree(self, tree, value, op, num_kids):
        self.assertIsInstance(tree, DecisionTree)
        self.assertIs(tree.value, value)
        self.assertIs(tree.op, op)
        self.assertIs(len(tree.kids), num_kids)

    def test_majority_value(self):
        binary_targets = [0, 0, 1, 0, 0, 1, 1, 1, 1]

        target_mode = majority_value_mode(binary_targets)
        expected_mode = 1

        self.assertTrue(expected_mode == target_mode)

    def test_same_value_true(self):
        x = [i / i for i in range(1, 100)]
        self.assertTrue(same_value(x))

    def test_same_value_false(self):
        x = range(1, 100)
        self.assertFalse(same_value(x))

    def test_decision_tree_learning_args_empty(self):
        examples = []
        attributes = range(45)
        binary_targets = []
        tree = decision_tree_learning(examples, attributes, binary_targets, majority_value_mode)
        self.assertTree(tree, 0, None, 0)

    def test_decision_tree_learning_same_values(self):
        examples = [range(i, 45 + i) for i in range(10)]
        attributes = range(45)
        binary_targets = [1 for i in range(10)]
        tree = decision_tree_learning(examples, attributes, binary_targets, majority_value_mode)
        self.assertTree(tree, 1, None, 0)

    def test_decision_tree_learning_no_attributes(self):
        examples = [range(i, 45 + i) for i in range(10)]
        attributes = []
        binary_targets = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
        tree = decision_tree_learning(examples, attributes, binary_targets, majority_value_mode)
        self.assertTree(tree, 1, None, 0)

    def test_decision_tree_learning_1(self):
        examples = [[0, 0],
                    [0, 1],
                    [1, 1],
                    [1, 1],
                    [1, 0]]
        attributes = [0, 1]
        binary_targets = [0, 1, 1, 1, 0]
        tree = decision_tree_learning(examples, attributes, binary_targets, majority_value_mode)
        self.assertTree(tree, None, 1, 2)
        self.assertTree(tree.kids[0], 0, None, 0)
        self.assertTree(tree.kids[1], 1, None, 0)

    def test_decision_tree_learning_2(self):
        examples = [[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]]
        attributes = [0, 1, 2]
        binary_targets = [1, 0, 1, 1, 1, 0, 1, 1]
        tree = decision_tree_learning(examples, attributes, binary_targets, majority_value_mode)
        self.assertTree(tree, None, 1, 2)
        self.assertTree(tree.kids[0], None, 2, 2)
        self.assertTree(tree.kids[0].kids[0], 1, None, 0)
        self.assertTree(tree.kids[0].kids[1], 0, None, 0)
        self.assertTree(tree.kids[1], 1, None, 0)


if __name__ == '__main__':
    unittest.main()
