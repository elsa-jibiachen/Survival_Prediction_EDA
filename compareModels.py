import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn import preprocessing

train_data = pd.read_csv('Assignment_train.csv')

test_df = pd.read_csv('Assignment_test.csv', header=None)
test_df.columns = ['pclass', 'age', 'gender', 'survived']

X_train = train_data.drop('survived', axis=1)
y_train = train_data['survived']

le_train = preprocessing.LabelEncoder()
X_train_encoded = X_train.apply(le_train.fit_transform)

X_test = test_df.drop('survived', axis=1)

le_test = preprocessing.LabelEncoder()
X_test_encoded = X_test.apply(le_test.fit_transform)

model_dt = DecisionTreeClassifier(criterion='entropy')
model_dt.fit(X_train_encoded, y_train)

y_pred_dt = model_dt.predict(X_test_encoded)

accuracy_dt = accuracy_score(test_df['survived'], y_pred_dt)
precision_dt = precision_score(test_df['survived'], y_pred_dt, pos_label='yes')
recall_dt = recall_score(test_df['survived'], y_pred_dt, pos_label='yes')
cm_dt = confusion_matrix(test_df['survived'], y_pred_dt)
print("Decision tree performance\n\tConfusion matrix:")
for row in cm_dt:
    print("\t\t", end="")
    print("\t".join(map(str, row)))
print(f"\tDecision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print(f"\tDecision Tree Precision: {precision_dt:.2f}")
print(f"\tDecision Tree Recall: {recall_dt:.2f}")

model_nb = GaussianNB()
model_nb.fit(X_train_encoded, y_train)

y_pred_nb = model_nb.predict(X_test_encoded)

accuracy_nb = accuracy_score(test_df['survived'], y_pred_nb)
precision_nb = precision_score(test_df['survived'], y_pred_nb, pos_label='yes')
recall_nb = recall_score(test_df['survived'], y_pred_nb, pos_label='yes')
cm_nb = confusion_matrix(test_df['survived'], y_pred_nb)
print("\nNaive Bayes classifier performance\n\tConfusion matrix:")
for row in cm_nb:
    print("\t\t", end="")
    print("\t".join(map(str, row)))
print(f"\tNaive Bayes Accuracy: {accuracy_nb * 100:.2f}%")
print(f"\tNaive Bayes Precision: {precision_nb:.2f}")
print(f"\tNaive Bayes Recall: {recall_nb:.2f}")
