import numpy as np
import pandas as pd

data = pd.read_csv("Assignment_train.csv", skiprows=0)

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        
        self.root = None
        
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:

            best_split = self.get_best_split(dataset, num_samples, num_features)

            if best_split["info_gain"]>0:

                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)

                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)

                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)

        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):

        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:

                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                    curr_info_gain = self.information_gain(y, left_y, right_y)

                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split
    
    def split(self, dataset, feature_index, threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child):
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
        
    def calculate_leaf_value(self, Y):
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
X_train = data.iloc[:, :-1].values
Y_train = data.iloc[:, -1].values.reshape(-1,1)
test_data = pd.read_csv("Assignment_test.csv", header=None)
X_test = test_data.iloc[:, :-1].values
Y_test = test_data.iloc[:, -1].values.reshape(-1,1)

classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=2)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test) 
Y_test_list = Y_test.tolist()
Y_test_flat = [item for sublist in Y_test_list for item in sublist]
data_confusion = pd.crosstab(Y_pred, Y_test_flat)
data_confusion.index.name = 'Predicted'
data_confusion.columns.name = 'Actual'

true_positives = data_confusion.loc['yes', 'yes']
true_negatives = data_confusion.loc['no', 'no']
false_positives = data_confusion.loc['yes', 'no']
false_negatives = data_confusion.loc['no', 'yes']

accuracy = (true_positives + true_negatives) / data_confusion.sum().sum()
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("Decision Tree classifier(from scratch) performance\n\tConfusion matrix:")
print("\t\t\tPredicted")
print("\t\tActual\tNo\tYes")
print("\t\tNo\t" + str(true_negatives) + "\t" + str(false_positives))
print("\t\tYes\t" + str(false_negatives) + "\t" + str(true_positives))
print(f"\tDecision Tree Accuracy: {accuracy * 100:.2f}%")
print(f"\tDecision Tree Precision: {precision:.2f}")
print(f"\tDecision Tree Recall: {recall:.2f}")