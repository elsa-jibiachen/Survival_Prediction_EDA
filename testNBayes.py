import numpy as np 
import pandas as pd
import json

def predict_example(row, lookup_table):
    
    class_estimates = np.array(lookup_table["prior"])
    for feature in row.index:

        try:
            value = row[feature]
            probabilities = lookup_table[feature].get(value, [0.0, 0.0])
            class_estimates = class_estimates * probabilities

        except KeyError:
            continue

    index_max_class = class_estimates.argmax()
    prediction = lookup_table["class_names"][index_max_class]
    
    return prediction

with open('probabilities.txt', 'r') as file:
    lookup_table = json.load(file)

df_test = pd.read_csv("Assignment_test.csv", header=None)
df_test.columns = ['pclass', 'age', 'gender', 'survived']

predictions = df_test.apply(predict_example, axis=1, args=(lookup_table,))

Y_test = df_test.iloc[:, -1].values.reshape(-1,1)

Y_test_list = Y_test.tolist()
Y_test_flat = [item for sublist in Y_test_list for item in sublist]

pred_list = list(predictions)

data_confusion = pd.crosstab(pred_list, Y_test_flat)
data_confusion.index.name = 'Predicted'
data_confusion.columns.name = 'Actual'

true_positives = data_confusion.loc['yes', 'yes']
true_negatives = data_confusion.loc['no', 'no']
false_positives = data_confusion.loc['yes', 'no']
false_negatives = data_confusion.loc['no', 'yes']

accuracy = (true_positives + true_negatives) / data_confusion.sum().sum()
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("Naive Bayes classifier(from scratch) performance\n\tConfusion matrix:")
print("\t\t\tPredicted")
print("\t\tActual\tNo\tYes")
print("\t\tNo\t" + str(true_negatives) + "\t" + str(false_positives))
print("\t\tYes\t" + str(false_negatives) + "\t" + str(true_positives))
print(f"\tNaive Bayes Accuracy: {accuracy * 100:.2f}%")
print(f"\tNaive Bayes Precision: {precision:.2f}")
print(f"\tNaive Bayes Recall: {recall:.2f}")