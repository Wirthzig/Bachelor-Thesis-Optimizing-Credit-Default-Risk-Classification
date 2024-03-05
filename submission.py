import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import models


def create_submission(train_file, test_file, applied_model, original=False):
    synthetic_train = pd.read_csv(train_file)
    synthetic_train = synthetic_train.rename(columns=lambda x: x.replace('-', '_').replace('/', '_').replace(':', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace(',', '_'))
    synthetic_train = synthetic_train.rename(columns=lambda x: x.replace('__', '_').replace('___', '_').replace('____', '_'))
    original_test = pd.read_csv(test_file)
    if original == True:
        #synthetic_train = synthetic_train.sample(frac=0.2, ignore_index=True)
        synthetic_train = pd.DataFrame(synthetic_train)
    print('Training shape: ', synthetic_train.shape)
    print('Testing shape: ', original_test.shape)
    print("Default ratio of train set: ", sum(synthetic_train["TARGET"]) / len(synthetic_train["TARGET"]))

    if applied_model == "LightGBM":
        submission = models.lgbm_basic(train=synthetic_train, n_folds=5,submission=True,test=original_test)
    elif applied_model == "Logistic Regression":
        submission = models.logistic_regression(train=synthetic_train, n_folds=5,submission=True,test=original_test)
    elif applied_model == "Neural Network":
        submission = models.neural_network(train=synthetic_train, n_folds=5,submission=True,test=original_test)
    else:
        submission = models.logistic_regression(train=synthetic_train, n_folds=5,submission=True,test=original_test)

    filename = os.path.basename(train_file)
    filename = os.path.splitext(filename)[0]
    name = "../Data/Submission Data/SUBMISSION_" + str(filename)+".csv"

    if original==True:
        name = "../Data/Submission Data/SUBMISSION_ORIGINAL_" + str(applied_model)+".csv"
    print("Created submission file: ", name)
    submission.columns = ["Id", "Probability"]
    submission.to_csv(name, index=False)
    print("----------------------------------------------------------------------------------------------------")
    return None

