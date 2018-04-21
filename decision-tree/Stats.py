def build_confusion_matrix(predictions, results):
    matrix = [[0 for _ in range(6)] for _ in range(6)]
    for index, prediction in enumerate(predictions):
        matrix[results[index] - 1][prediction - 1] += 1
    return matrix


def combine_confusion_matrices(a, b):
    c = [[0 for _ in range(6)] for _ in range(6)]
    for i in range(6):
        for j in range(6):
            c[i][j] = a[i][j] + b[i][j]
    return c


def visualise_confusion_matrix(matrix):
    line = ""
    line += "       Predicted\n"
    line += "Actual"
    for i in range(len(matrix[0])):
        line += " " + str(i + 1)
    line += "\n"
    for i in range(len(matrix)):
        line += "     " + str(i + 1)
        for j in range(len(matrix)):
            line += " " + str(matrix[i][j])
        line += "\n"
    print(line)


def get_emotion_stats(confusion_matrix):
    emotion_stats = [[0 for _ in range(4)] for _ in range(6)]
    # for each emotion (TP, TN, FP, FN)
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            if i == j:
                emotion_stats[i][0] += confusion_matrix[i][j]
            elif i != j:
                emotion_stats[i][3] += confusion_matrix[i][j]
                emotion_stats[j][2] += confusion_matrix[i][j]
            # for everything that is not i or j
            # TN += confusion_matrix[i][j]
            for e in range(len(confusion_matrix)):
                if e != i and e != j:
                    emotion_stats[e][1] += confusion_matrix[i][j]
    return emotion_stats


def get_recall_precision_f_measure(emotion_stat, alpha=1):
    TP, TN, FP, FN = emotion_stat
    recall = (TP / (TP + FN)) * 100
    precision = (TP / (TP + FP)) * 100
    f_measure = (1 + alpha) * ((precision * recall) / ((alpha * precision) + recall))
    return recall, precision, f_measure


def get_emotion_recall_precision_f_measure(emotion_stats):
    return list(map(lambda emotion_stat: get_recall_precision_f_measure(emotion_stat), emotion_stats))


def get_stats_from_confusion_matrix(matrix):
    return get_emotion_recall_precision_f_measure(get_emotion_stats(matrix))
