import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Import Data
print("Importing Raw Data...")
train = pd.read_csv("../Data/GiveMeSomeCredit/cs-training.csv")
test = pd.read_csv("../Data/GiveMeSomeCredit/cs-test.csv")
print("Done...\n")


# Customize colors
background_color = '#FFFFFF'
line_color = '#1CD75F'
dot_color = '#1CD75F'
label_color = "#808080"  # "#121212"
transparent = "#5F5F5F00"
grey_color = "#b3b3b3"#"#808080"

blue="#0000a7"#"#30EEE8"
yellow="#008176"#"#EEDA30"
green=blue#"#1CD75F"
red= "#c1272d"#EE304E"
pink="#EA30EE"
orange= red#"#EE7B30"


def EDA(train, test):
    # Adjust the Pandas display settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # Print out the number of rows and columns for each dataset
    print(f"Train dataset: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"Test dataset: {test.shape[0]} rows, {test.shape[1]} columns")

    # Then, generate descriptive statistics for each column in the datasets
    print("\nTrain dataset summary:")
    print(train.describe(include='all'))
    print("\nTest dataset summary:")
    print(test.describe(include='all'))

    # Plot for target variable
    plot_target_distribution(train)

    # Plots for missing values
    missing_values_plot(train, test)


def plot_target_distribution(train):
    fig, ax = plt.subplots()
    default = sum(train["TARGET"])
    non_default = len(train) - default
    ax.bar('Non-Default', non_default, color=green, edgecolor="white", label='Non-Default')
    ax.bar('Default', default, color=red, edgecolor="white", label='Default')
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    ax.set_title('Distribution of TARGET Variable', color=label_color)
    ax.tick_params(axis='x', colors=label_color)
    ax.tick_params(axis='y', colors=label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(transparent)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('../Data/Outputs/EDA/', 'target_distribution.png'), dpi=300)
    print("Created Plot: ", 'target_distribution.png')





def missing_values_plot(train, test):
    train = train[["MonthlyIncome", "NumberOfDependents"]]
    test = test[["MonthlyIncome", "NumberOfDependents"]]
    missing_values_train = (train.isnull().sum() / len(train)) * 100
    missing_values_test = (test.isnull().sum() / len(test)) * 100

    # Filter out columns that don't contain any missing values
    missing_values_train = missing_values_train.dropna()
    missing_values_test = missing_values_test.dropna()
    print(missing_values_train)
    missing_values_df = pd.DataFrame({'Train': missing_values_train, 'Test': missing_values_test})
    missing_values_df = missing_values_df.sort_values(by='Train', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    missing_values_df.plot(kind='bar', ax=ax, color=[grey_color, label_color, grey_color, label_color], edgecolor="white")
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    ax.set_title('Percentage of Missing Values in Each Column', color=label_color)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Missing Values (%)')
    ax.tick_params(axis='x', colors=label_color)
    ax.tick_params(axis='y', colors=label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(transparent)
    ax.legend(loc="upper right", labelcolor = label_color)
    plt.tight_layout()
    plt.savefig(os.path.join('../Data/Outputs/EDA/', 'missing_values.png'), dpi=900)
    print("Created Plot: ",'missing_values.png')


EDA(train,test)


































#---------------------------------------------------------------------------------------------------------
# Preprocess Data
print("Preprocessing Data...")
print("OUTLIER!")
train = train.loc[train["DebtRatio"] <= train["DebtRatio"].quantile(0.95)]
train = train.loc[(train["RevolvingUtilizationOfUnsecuredLines"] >= 0) & (train["RevolvingUtilizationOfUnsecuredLines"] < 13)]
train = train.loc[train["NumberOfTimes90DaysLate"] <= 17]


print("IMPUTE!")
dependents_mode = train["NumberOfDependents"].mode()[0] # impute with mode
train["NumberOfDependents"] = train["NumberOfDependents"].fillna(dependents_mode)
income_median = train["MonthlyIncome"].median()
train["MonthlyIncome"] = train["MonthlyIncome"].fillna(income_median)

dependents_mode = test["NumberOfDependents"].mode()[0] # impute with mode
test["NumberOfDependents"] = test["NumberOfDependents"].fillna(dependents_mode)
income_median = test["MonthlyIncome"].median()
test["MonthlyIncome"] = test["MonthlyIncome"].fillna(income_median)
print("Done...\n")

# Save Essential Items
train.to_csv("../Data/GiveMeSomeCredit/cs-training-preprocessed.csv", index=False)
test.to_csv("../Data/GiveMeSomeCredit/cs-testing-preprocessed.csv", index=False)

print("Completed!")
#---------------------------------------------------------------------------------------------------------

EDA(train,test)