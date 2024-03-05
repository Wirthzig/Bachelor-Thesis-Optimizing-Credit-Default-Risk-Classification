import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


def lgbm_basic(train, n_folds=5, submission=False, test=None):
    labels = train['TARGET'].values
    features = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    features = pd.get_dummies(features)
    cat_indices = 'auto'
    features = np.array(features)

    if submission and test is None:
        raise ValueError("Parameter 'required_if_true' is required when 'condition' is True.")
    if submission:
        test_ids = test['SK_ID_CURR']
        test = test.drop(columns=['SK_ID_CURR'])
        test_features = np.array(test)
        k_fold = KFold(n_splits=n_folds, shuffle=True)
        test_predictions = np.zeros(test_features.shape[0])
        for train_indices, valid_indices in k_fold.split(features):
            train_features, train_labels = features[train_indices], labels[train_indices]
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
            model = lgb.LGBMClassifier(n_estimators=100, objective='binary', boosting_type='gbdt', learning_rate=0.05,
                                       reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)
            model.fit(train_features, train_labels, eval_metric='auc',
                      eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                      eval_names=['valid', 'train'], categorical_feature=cat_indices, early_stopping_rounds=100, verbose=10000)
            best_iteration = model.best_iteration_
            test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
        return submission

    valid_accuracies = []
    valid_precisions = []
    valid_recalls = []
    valid_f1s = []
    valid_aucs = []
    valid_kappas = []
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        model = lgb.LGBMClassifier(n_estimators=100, objective='binary', boosting_type='gbdt', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1)
        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices, early_stopping_rounds=100, verbose=10000)
        valid_preds = model.predict(valid_features)
        valid_probs = model.predict_proba(valid_features)[:, 1]
        accuracy = accuracy_score(valid_labels, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, valid_preds, average='binary')
        auc = roc_auc_score(valid_labels, valid_probs)
        kappa = cohen_kappa_score(valid_labels, valid_preds)
        valid_accuracies.append(accuracy)
        valid_precisions.append(precision)
        valid_recalls.append(recall)
        valid_f1s.append(f1)
        valid_aucs.append(auc)
        valid_kappas.append(kappa)
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    avg_accuracy = sum(valid_accuracies) / len(valid_accuracies)
    avg_precision = sum(valid_precisions) / len(valid_precisions)
    avg_recall = sum(valid_recalls) / len(valid_recalls)
    avg_f1 = sum(valid_f1s) / len(valid_f1s)
    avg_auc = sum(valid_aucs) / len(valid_aucs)
    avg_kappa = sum(valid_kappas) / len(valid_kappas)
    average_valid_score = (avg_accuracy + avg_precision + avg_recall + avg_f1 + avg_auc + avg_kappa) / 6
    return average_valid_score, avg_auc


def logistic_regression(train, n_folds=5, submission=False, test=None):
    labels = train['TARGET'].values
    features = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    features = np.array(features)

    if submission and test is None:
        raise ValueError("Parameter 'required_if_true' is required when 'condition' is True.")
    if submission:
        test_ids = test['SK_ID_CURR']
        test = test.drop(columns=['SK_ID_CURR'])
        test_features = np.array(test)
        k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)
        test_predictions = np.zeros(test_features.shape[0])
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        test_features = scaler.transform(test_features)
        features = np.array(features)
        test_features = np.array(test_features)
        for train_indices, valid_indices in k_fold.split(features):
            train_features, train_labels = features[train_indices], labels[train_indices]
            model = LogisticRegression(C=0.01, n_jobs=-1, random_state=50)
            model.fit(train_features, train_labels)
            test_predictions += model.predict_proba(test_features)[:, 1] / k_fold.n_splits
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
        return submission

    valid_accuracies = []
    valid_precisions = []
    valid_recalls = []
    valid_f1s = []
    valid_aucs = []
    valid_kappas = []
    train_aucs = []  # new list to store training AUC scores

    k_fold = KFold(n_splits=n_folds, shuffle=True)
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        model = LogisticRegression(C=0.01)
        model.fit(train_features, train_labels)
        train_probs = model.predict_proba(train_features)[:, 1]  # predict probabilities for the training data
        train_auc = roc_auc_score(train_labels, train_probs)  # calculate training AUC score
        valid_preds = model.predict(valid_features)
        valid_probs = model.predict_proba(valid_features)[:, 1]
        accuracy = accuracy_score(valid_labels, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, valid_preds, average='binary')
        auc = roc_auc_score(valid_labels, valid_probs)
        kappa = cohen_kappa_score(valid_labels, valid_preds)
        train_aucs.append(train_auc)  # add training AUC to list
        valid_accuracies.append(accuracy)
        valid_precisions.append(precision)
        valid_recalls.append(recall)
        valid_f1s.append(f1)
        valid_aucs.append(auc)
        valid_kappas.append(kappa)
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    avg_accuracy = sum(valid_accuracies) / len(valid_accuracies)
    avg_precision = sum(valid_precisions) / len(valid_precisions)
    avg_recall = sum(valid_recalls) / len(valid_recalls)
    avg_f1 = sum(valid_f1s) / len(valid_f1s)
    avg_auc = sum(valid_aucs) / len(valid_aucs)
    avg_kappa = sum(valid_kappas) / len(valid_kappas)
    avg_train_auc = sum(train_aucs) / len(train_aucs)  # calculate average training AUC
    auc_diff = abs(avg_train_auc - avg_auc)  # calculate the absolute difference between train and validation AUCs
    average_valid_score = (avg_precision + avg_recall + avg_f1 + auc_diff + avg_kappa) / 5  # replace avg_auc with auc_diff
    return average_valid_score, avg_auc


def neural_network(train, n_folds=5, submission=False, test=None):

    if submission and test is None:
        raise ValueError("Parameter 'required_if_true' is required when 'condition' is True.")
    if submission:
        # Extract the ids
        train_ids = train['SK_ID_CURR']
        test_ids = test['SK_ID_CURR']
        # Extract the labels for training
        labels = train['TARGET']
        # Remove the ids and target
        features = train.drop(columns=['SK_ID_CURR', 'TARGET'])
        test_features = test.drop(columns=['SK_ID_CURR'])
        # One Hot Encoding
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join='inner', axis=1)


        # Convert to np arrays
        features = np.array(features)
        test_features = np.array(test_features)
        # Create the kfold object
        k_fold = KFold(n_splits=n_folds, shuffle=True)
        # Empty array for test predictions
        test_predictions = np.zeros(test_features.shape[0])
        # Empty array for out of fold validation predictions
        out_of_fold = np.zeros(features.shape[0])

        # Standardization for logistic regression
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        test_features = scaler.transform(test_features)
        # Iterate through each fold
        for train_indices, valid_indices in k_fold.split(features, labels):
            # Training data for the fold
            train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
            # Create the model
            model = Sequential()
            model.add(Dense(64, input_dim=train_features.shape[1], activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['AUC'])
            # Train the model
            model.fit(train_features, train_labels, epochs=10, verbose=0)
            # Make predictions
            test_predictions += model.predict(test_features).flatten() / k_fold.n_splits
        # Make the submission dataframe
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
        return submission

    labels = train['TARGET'].values
    features = train.drop(columns=['SK_ID_CURR', 'TARGET'])
    features = np.array(features)

    valid_accuracies = []
    valid_precisions = []
    valid_recalls = []
    valid_f1s = []
    valid_aucs = []
    valid_kappas = []
    k_fold = KFold(n_splits=n_folds, shuffle=True)
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        model = Sequential()
        model.add(Dense(64, input_dim=train_features.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['AUC'])
        model.fit(train_features, train_labels, epochs=10, verbose=0)
        valid_probs = model.predict(valid_features, verbose=0).flatten()
        valid_preds = np.where(valid_probs > 0.5, 1, 0)
        accuracy = accuracy_score(valid_labels, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, valid_preds, average='binary')
        auc = roc_auc_score(valid_labels, valid_probs)
        kappa = cohen_kappa_score(valid_labels, valid_preds)
        valid_accuracies.append(accuracy)
        valid_precisions.append(precision)
        valid_recalls.append(recall)
        valid_f1s.append(f1)
        valid_aucs.append(auc)
        valid_kappas.append(kappa)
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    avg_accuracy = sum(valid_accuracies) / len(valid_accuracies)
    avg_precision = sum(valid_precisions) / len(valid_precisions)
    avg_recall = sum(valid_recalls) / len(valid_recalls)
    avg_f1 = sum(valid_f1s) / len(valid_f1s)
    avg_auc = sum(valid_aucs) / len(valid_aucs)
    avg_kappa = sum(valid_kappas) / len(valid_kappas)
    average_valid_score = (avg_accuracy + avg_precision + avg_recall + avg_f1 + avg_auc + avg_kappa) / 6
    return average_valid_score, avg_auc


