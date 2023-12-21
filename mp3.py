import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

np.random.seed(0)

def compute_metric(labels, expected):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }

def mlpTraining(X, Y):
    # Grid Search:
    # param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
    #     'alpha': [0.00001, 0.0000001, 0.000000001],
    # }

    # # Create the MLPClassifier
    # mlp = MLPClassifier(max_iter=1000)  # You can adjust max_iter based on convergence

    # # Create the GridSearchCV object
    # grid_search = GridSearchCV(mlp, param_grid, scoring='f1', cv=5)

    # # Fit the grid search to the data
    # grid_search.fit(X, Y)

    # # Get the best hyperparameters and the corresponding model
    # best_params = grid_search.best_params_
    # print(best_params)
    # best_mlp = grid_search.best_estimator_
    
    best_params = {
        'hidden_layer_sizes': (100, 50),
        'alpha': 0.0000001,
        'learning_rate': 'adaptive',
        'activation': 'relu',
        'solver': 'adam',
        # 'batch_size': 200,
    }
    best_mlp = MLPClassifier(max_iter=1000)
    best_mlp.set_params(**best_params)
    best_mlp.fit(X, Y)
    return best_mlp

def preprocess(data):
    # Preprocessing
    # if 'target' in data.columns:
    #     X = data.drop('target', axis=1)
    # else:
    #     X = data
    
    X = data

    unnecessary_columns = ['QUANTIZED_INC', 'QUANTIZED_AGE', 'QUANTIZED_WORK_YEAR'] 
    X = X.drop(unnecessary_columns, axis=1)
    
    
    # scaler_float = RobustScaler()
    # scaler_int = MinMaxScaler()
    # numF_columns = X.select_dtypes(include=['float64']).columns
    # numI_columns = X.select_dtypes(include=['int64']).columns
    # # print(X.dtypes)
    # scaler_float.fit(X[numF_columns])
    # X[numF_columns] = scaler_float.transform(X[numF_columns])
    # scaler_int.fit(X[numI_columns])
    # X[numI_columns] = scaler_int.transform(X[numI_columns])
    
    # scaler = RobustScaler()
    scaler = StandardScaler()
    X[['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']] = scaler.fit_transform(X[['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']])
    # X[['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']] = scaler.fit_transform(X[['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']])
    
    categorical_columns = X.select_dtypes(include=['object']).columns
    # categorical_columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
    # categorical_columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'QUANTIZED_INC', 'QUANTIZED_AGE', 'QUANTIZED_WORK_YEAR']
    # categorical_columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    
    # encoder = OneHotEncoder(sparse=False)
    # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # encoded_columns = pd.DataFrame(encoder.fit_transform(X[categorical_columns]), columns=encoder.get_feature_names_out(X[categorical_columns].columns))
    
    encoder = pd.get_dummies(X[categorical_columns], drop_first=True)
    encoded_columns = pd.concat([X, encoder], axis=1)
    
    X_encoded = pd.concat([X, encoded_columns], axis=1)
    X_encoded = X_encoded.drop(categorical_columns, axis=1)
    
    return X_encoded

def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: 
        testing_data: the same as training_data with "target" removed.

    Output:
        testing_prediction: List[int]
    Example output:
    return random.choices([0, 1, 2], k=len(testing_data))
    """
    predict = np.zeros(len(testing_data))
    
    # Preprocessing:
    X_train = preprocess(training_data)
    X_test = preprocess(testing_data)
    # print(len(X_test.columns))
    extra_columns_in_X_train = set(X_train.columns) - set(X_test.columns)
    X_train = X_train.drop(extra_columns_in_X_train, axis=1)
    # X_train = X_train.values
    # Y_train = training_data['target'].values
    Y_train = training_data['target']
    # X_test = X_test.values
    
    best_mlp = mlpTraining(X_train, Y_train)
    
    # print(type(best_mlp.predict(X_test)))
    # print(best_mlp.predict(X_test).shape)
    # print(best_mlp.predict(X_test))
    predict = best_mlp.predict(X_test)
    # predict = best_mlp.predict(X_train)

    return predict


if __name__ == '__main__':

    training = pd.read_csv('./data/train.csv')
    development = pd.read_csv('./data/dev.csv')

    target_label = development['target']
    development.drop('target', axis=1, inplace=True)
    prediction = run_train_test(training, development)
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    # status = compute_metric(prediction, training['target'].values)
    print(status)