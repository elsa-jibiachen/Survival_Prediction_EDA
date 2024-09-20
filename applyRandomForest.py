import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model_rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(estimator=model_rf, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42)
random_search.fit(X_train_encoded, y_train)

best_params = random_search.best_params_

best_model_rf = RandomForestClassifier(random_state=42, **best_params)
best_model_rf.fit(X_train_encoded, y_train)

y_pred_rf = best_model_rf.predict(X_test_encoded)

accuracy = accuracy_score(test_df['survived'], y_pred_rf)
precision = precision_score(test_df['survived'], y_pred_rf, pos_label='yes')
recall = recall_score(test_df['survived'], y_pred_rf, pos_label='yes')
cm = confusion_matrix(test_df['survived'], y_pred_rf)
print("Random Forest classifier performance\n\tConfusion matrix:")
for row in cm:
    print("\t\t", end="")
    print("\t".join(map(str, row)))
print(f"\tRandom Forest Accuracy: {accuracy * 100:.2f}%")
print(f"\tRandom Forest Precision: {precision:.2f}")
print(f"\tRandom Forest Recall: {recall:.2f}")
print("\tBest Parameters:", best_params)
