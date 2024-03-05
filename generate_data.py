import warnings
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import generate_data
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
import pandas as pd
import time

################################################
train_file = "../Data/GiveMeSomeCredit/cs-training-preprocessed.csv"
test_file = "../Data/GiveMeSomeCredit/cs-testing-preprocessed.csv"
################################################

print("Read in data...")
global real_train
global real_test
real_train = pd.read_csv(train_file)
real_train = real_train.rename(columns=lambda x: x.replace('-', '_').replace('/', '_').replace(':', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace(',', '_'))
real_train = real_train.rename(columns=lambda x: x.replace('__', '_').replace('___', '_').replace('____', '_'))

y = generate_data.real_train['TARGET']
X = generate_data.real_train.drop('TARGET', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
real_train = pd.DataFrame(X_train)
real_test = pd.DataFrame(X_test)
real_train['TARGET'] = list(y_train)
real_test['TARGET'] = list(y_test)
print("Finished")
print('Training shape: ', real_train.shape)
print("Default ratio of train set: ", sum(real_train["TARGET"]) / len(real_train["TARGET"]))

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def generate_synthetic_data(sampling_strategy, k_neighbors, smote_kind):
    # Map the smote_kind parameter to the corresponding SMOTE variant
    smote_variants = {
        0: SMOTE,
        1: BorderlineSMOTE,
        2: SMOTEENN,
        3: SMOTETomek,
    }
    smote_variant = smote_variants[int(smote_kind)]
    y_train = generate_data.real_train['TARGET']
    X_train = generate_data.real_train.drop('TARGET', axis=1)
    X_train = pd.DataFrame(X_train)
    # Instantiate the SMOTE object with the input parameters
    if smote_variant in [SMOTE, BorderlineSMOTE]:
        smote = smote_variant(sampling_strategy=sampling_strategy, k_neighbors=int(k_neighbors))
    elif smote_variant in [SMOTEENN, SMOTETomek]:
        smote = smote_variant(sampling_strategy=sampling_strategy)
    # Generate the synthetic data using SMOTE
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = list(y_resampled)
    # Combine the resampled features and labels into a single DataFrame
    synthetic_data = X_resampled.copy()
    synthetic_data["TARGET"] = y_resampled
    col_names = list(synthetic_data.columns)
    synthetic_data.columns = col_names


    train = generate_data.real_train.copy()
    #train = train.sample(frac=0.2)

    return pd.DataFrame(synthetic_data), pd.DataFrame(train)

