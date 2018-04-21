from ete3 import Tree


def tree_string(tree, from_value=None):
    preced = str(from_value) + ", " if from_value is not None else ""
    if len(tree.kids) == 0:
        return "\'" + preced + str(tree.value) + '\''
    else:
        result = "("
        for i in range(len(tree.kids)):
            result += tree_string(tree.kids[i], i)
            if i < len(tree.kids) - 1:
                result += ","
        result += ")\'" + preced + "AU" + str(tree.op) + '\''
        return result


def visualize_ete_tree(tree):
    string = tree_string(tree) + ";"
    print(string)
    t = Tree(tree_string(tree) + ";", format=1, quoted_node_names=1)
    for node in t.traverse("levelorder"):
        if not node.is_leaf():
            node.add_features(op=node.name)
    print(t.get_ascii(show_internal=True))  # attributes=['name', 'op'],

# visualize_ete_tree(DecisionTree(None, 0, [DecisionTree(1, None, []), DecisionTree(0, 2, [DecisionTree(0, None, []), DecisionTree(1, None, [])])]))
