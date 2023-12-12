"""data.py

load & process OpenML datasets
"""

import pandas as pd
import numpy as np
import openml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

openml_df_original = openml.datasets.list_datasets(output_format="dataframe")

openml_df = openml_df_original[openml_df_original["NumberOfNumericFeatures"] > 0]
openml_df.dropna(inplace=True)  # drop all the rows with missing entries
openml_df = openml_df[openml_df["NumberOfInstancesWithMissingValues"] == 0]
openml_df = openml_df[openml_df["NumberOfMissingValues"] == 0]
openml_df = openml_df[openml_df["NumberOfFeatures"] <= 20]
openml_df = openml_df[openml_df["NumberOfClasses"] <= 2]
openml_df = openml_df[openml_df["NumberOfInstances"] <= 10000]
openml_df = openml_df[openml_df["NumberOfInstances"] >= 1000]
openml_df = openml_df[openml_df["NumberOfNumericFeatures"] >= 5]
openml_df = openml_df[openml_df["MinorityClassSize"] >= openml_df["MajorityClassSize"] / 2]


openml_datasets_train = {}
openml_datasets_test = {}
good_dataset_list = map(int, list(openml_df.did.values))
counter = 0
max_num_class = -1
exists_dataset_name = []
for i in good_dataset_list:
    try:
        openml.datasets.check_datasets_active([i])
    except:
        continue
    ds = openml.datasets.get_dataset(i, download_data=False, download_qualities=False, download_features_meta_data=False)  # load the openML-i dataset
    if ds.name in exists_dataset_name or 'fri' in ds.name:
        continue
    else:
        exists_dataset_name.append(ds.name)
    try:
        X, y, categorical_indicator, _ = ds.get_data(target=ds.default_target_attribute)
    except:
        print('Dataset {} error'.format(i))
        continue
    if y.dtype.name != "category":
        # print('Not a category series')
        continue
    counter += 1
    non_categorial_indices = np.where(np.array(categorical_indicator) == False)[0]  # find where categorical columns are
    Xy = pd.concat([X, y], axis=1, ignore_index=True)  # X & y concatenated together
    Xy = Xy.iloc[:,
         [*non_categorial_indices, -1]]  # Slice columns -- ignore categorical X columns and add y column (-1)
    Xy.replace('?', np.NaN, inplace=True)  # replace ? with NaNs
    Xy = Xy[Xy.iloc[:, -1].notna()]  # remove all the rows whose labels are NaN
    y_after_NaN_removal = Xy.iloc[:, -1]
    Xy.dropna(axis=1, inplace=True)  # drop all the columns with missing entries
    Xy.dropna(inplace=True)  # drop all the rows with missing entries
    assert ((Xy.iloc[:, -1] == y_after_NaN_removal).all())
    X, y = Xy.iloc[:, :-1], Xy.iloc[:, -1]

    if X.shape[0] == 0 or X.shape[1] == 0:  # check if X is empty or not
        continue
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = y.cat.codes.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)

        openml_datasets_train[i] = {"X": X_train, "y": y_train}
        print(f"# {counter}, dataset id={i}, name={ds.name}, class={np.unique(y_train)}, train_X={X_train.shape}, train_y={y_train.shape}")
        if np.max(np.unique(y_train)) > max_num_class:
            max_num_class = np.max(np.unique(y_train))
        openml_datasets_test[i] = {"X": X_test, "y": y_test}

root = 'data/'
if not os.path.exists(root):
    os.makedirs(root)

print("max number of class:", max_num_class + 1)

import pickle
with open(root + 'openml_train2.npy', 'wb') as f:
    pickle.dump(openml_datasets_train, f)
with open(root + 'openml_test2.npy', 'wb') as f:
    pickle.dump(openml_datasets_test, f)
