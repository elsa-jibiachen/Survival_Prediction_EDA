import pandas as pd
import json
from pprint import pprint

df_train = pd.read_csv("Assignment_train.csv", skiprows=0)

def create_table(df, label_column):
    table = {}

    value_counts = df[label_column].value_counts().sort_index()
    table["class_names"] = value_counts.index.tolist()
    table["prior"] = (value_counts.values/len(df)).tolist()

    for feature in df.drop(label_column, axis=1).columns:
        table[feature] = {}

        counts = df.groupby(label_column)[feature].value_counts()
        df_counts = counts.unstack(label_column)

        if df_counts.isna().any(axis=None):
            df_counts.fillna(value=0, inplace=True)
            df_counts += 1

        df_probabilities = df_counts / df_counts.sum()
        for value in df_probabilities.index:
            probabilities = df_probabilities.loc[value].tolist()
            table[feature][value] = probabilities
            
    return table

lookup_table = create_table(df_train, label_column="survived")

with open('probabilities.txt', 'w') as file:
    json.dump(lookup_table, file)