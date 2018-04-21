import pickle
import scipy.io as sio
import sys
from EmotionRecognition import DecisionTree, test_trees, calculate_classifier_error

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Pass path to data file as command argument e.g. python3 AssessorTesting.py "
              "Data/cleandata_students.mat")
    else:
        mat = sio.loadmat(sys.argv[1])
        x = mat['x']
        y = list(map(lambda f: f[0], mat['y']))
        clean_data_trees = [pickle.load(open("saved_clean_data_pkl_trees/tree{}.pkl".format(t + 1), "rb")) for t in range(6)]
        predictions = test_trees(clean_data_trees, x)
        print("The trees trained on the clean data set made these predictions on this data set:")
        print(predictions)
        error = calculate_classifier_error(predictions, y)
        print("The classifier error for this data is: " + str(error))
        print("This is a classification rate of: " + str(1.0 - error))
