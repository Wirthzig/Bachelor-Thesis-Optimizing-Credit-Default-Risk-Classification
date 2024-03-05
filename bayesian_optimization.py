import os
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import simple_colors
from bayes_opt import BayesianOptimization, UtilityFunction
import generate_data
import models
import pandas as pd


def update_performance(filename, model, sampling_strategy, k_neighbors, smote_kind, performance, auc, original_performance, original_auc, imbalance):
    new_row = {"Filename": filename, "Model": model, "Sampling Strategy": sampling_strategy, "K-Nearest Neighbors": k_neighbors, "Smote Kind": smote_kind, "Performance": performance, "AUC": auc, "OG_Performance": original_performance, "OG_AUC": original_auc, "Imbalance": imbalance}
    df = pd.read_csv("../Data/performances.csv")
    df = df.append(new_row, ignore_index=True)
    df.to_csv("../Data/performances.csv", index=False)


def increment_filename(path, name):
    base_filename = os.path.splitext(name)[0]
    max_count = 0
    for file in os.listdir(path):
        if file == ".DS_Store":
            continue
        base_file = os.path.splitext(file)[0]
        parts = base_file.split("_")
        if parts[2] == name:
            try:
                file_number = int(parts[-1])
                max_count = max(max_count, file_number + 1)
            except ValueError:
                pass
    new_filename = f"{path}SYNTHETIC_DATA_{base_filename}_{max_count}.csv"
    return new_filename


def objective_function_lgbm_basic(sampling_strategy, k_neighbors, smote_kind):
    synthetic_data, _ = generate_data.generate_synthetic_data(sampling_strategy, k_neighbors, smote_kind)
    performance, auc = models.lgbm_basic(train=synthetic_data, n_folds=5, submission=False, test=None)
    return performance


def objective_function_logistic_regression(sampling_strategy, k_neighbors, smote_kind):
    synthetic_data, _ = generate_data.generate_synthetic_data(sampling_strategy, k_neighbors, smote_kind)
    performance, auc = models.logistic_regression(train=synthetic_data, n_folds=5, submission=False, test=None)
    return performance


def objective_function_NN(sampling_strategy, k_neighbors, smote_kind):
    synthetic_data, _ = generate_data.generate_synthetic_data(sampling_strategy, k_neighbors, smote_kind)
    performance, auc = models.neural_network(train=synthetic_data, n_folds=5, submission=False, test=None)
    return performance


def optimize_Random_Search(model, init_points=5, n_iter=20):
    # Define the bounds of the parameters
    param_distributions = {
        'sampling_strategy': (0.1, 0.9),
        'k_neighbors': (2, 30),
        'smote_kind': (0, 3),
    }
    if model == "LightGBM":
        objective_function = objective_function_lgbm_basic
        chosen_model = models.lgbm_basic
    elif model == "Logistic Regression":
        objective_function = objective_function_logistic_regression
        chosen_model = models.logistic_regression
    elif model == "Neural Network":
        objective_function = objective_function_NN
        chosen_model = models.neural_network
    else:
        objective_function = objective_function_logistic_regression
        chosen_model = models.logistic_regression

    sampling_strategy_distribution = param_distributions['sampling_strategy']
    k_neighbors_distribution = param_distributions['k_neighbors']
    smote_kind_distribution = param_distributions['smote_kind']
    best_score = None
    best_params = None
    objective_values = []
    print("\n")
    print("---------------------------- Running Optimization --------------------------------")
    print("|    iter    |   target  |   sampling_strategy   |   k_neighbors |   smote_kind  |")
    params = []
    for i in range(int(n_iter + init_points)):
        sampling_strategy = np.random.uniform(*sampling_strategy_distribution)
        k_neighbors = np.random.randint(*k_neighbors_distribution)
        smote_kind = np.random.randint(*smote_kind_distribution)
        params = [sampling_strategy, k_neighbors, smote_kind]
        score = objective_function(sampling_strategy=params[0], k_neighbors=params[1], smote_kind=params[2])
        objective_values.append(score)
        sampling_strategy = round(sampling_strategy,4)
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            progress = f"|      {i}       |   {round(score,4)}  |     {sampling_strategy}     |       {k_neighbors}     |       {smote_kind}      |"
            print(simple_colors.blue(progress))
        else:
            progress = f"|      {i}       |   {round(score,4)}  |     {sampling_strategy}     |       {k_neighbors}     |       {smote_kind}      |"
            print(progress)

    df = pd.read_csv("../Data/process.csv")
    if len(df) < len(objective_values):
        missing_rows = len(objective_values) - len(df)
        for _ in range(missing_rows):
            df = df.append(pd.Series(), ignore_index=True)
    df[str("RANDOM" + model)] = pd.Series(objective_values)
    df.to_csv("../Data/process.csv", index=False)

    sampling_strategy = best_params[0]
    k_neighbors = best_params[1]
    smote_kind = best_params[2]
    optimized_synthetic_data, original_data = generate_data.generate_synthetic_data(sampling_strategy=sampling_strategy,
                                                                                    k_neighbors=k_neighbors,
                                                                                    smote_kind=smote_kind)
    synthetic_performance, synthetic_auc = chosen_model(optimized_synthetic_data)
    print("Model performance on the synthetic data: ", f" | AUC-score: {synthetic_auc:.4f} | Overall Performance: {synthetic_performance:.4f} |")
    original_performance, original_auc = chosen_model(original_data)
    print("Model performance on the original data: ", f" | AUC-score: {original_auc:.4f} | Overall Performance: {original_performance:.4f} |")

    tempname = "RANDOM" + model
    name = increment_filename(path="../Data/Synthetic Data/", name=tempname)
    imbalance = sum(optimized_synthetic_data["TARGET"]) / len(optimized_synthetic_data["TARGET"])
    update_performance(filename=name, model=model, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, smote_kind=smote_kind, performance=synthetic_performance, auc=synthetic_auc, original_performance=original_performance, original_auc=original_auc, imbalance=imbalance)
    print("Create Synthetic Data for submission...")
    optimized_synthetic_data, _ = generate_data.generate_synthetic_data(sampling_strategy=sampling_strategy,
                                                                        k_neighbors=k_neighbors,
                                                                        smote_kind=smote_kind)
    optimized_synthetic_data.to_csv(name, index=False)
    return name


def optimize(model, init_points=5, n_iter=35):
    param_bounds = {
        'sampling_strategy': (0.1, 0.9),
        'k_neighbors': (2, 20),
        'smote_kind': (0, 3),
    }
    if model == "LightGBM":
        objective_function = objective_function_lgbm_basic
        chosen_model = models.lgbm_basic
    elif model == "Logistic Regression":
        objective_function = objective_function_logistic_regression
        chosen_model = models.logistic_regression
    elif model == "Neural Network":
        objective_function = objective_function_NN
        chosen_model = models.neural_network
    else:
        objective_function = objective_function_lgbm_basic
        chosen_model = models.logistic_regression
    optimizer = BayesianOptimization(f=None, pbounds=param_bounds, verbose=2)
    # utility function) to be the upper confidence bounds "ucb".
    # We set kappa = 1.96 to balance exploration vs exploitation.
    # xi = 0.01 is another hyper parameter which is required in the
    # arguments, but is not used by "ucb". Other acquisition functions
    # such as the expected improvement "ei" will be affected by xi.
    # utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)
    utility = UtilityFunction(kind="ucb")

    best_target = None
    print("\n")
    print("---------------------------- Running Optimization --------------------------------")
    print("|    iter    |   target  |   sampling_strategy   |   k_neighbors |   smote_kind  |")
    for i in range(init_points + n_iter):
        next_point = optimizer.suggest(utility)
        next_point["smote_kind"] = int(next_point["smote_kind"])
        next_point["k_neighbors"] = int(next_point["k_neighbors"])

        target = objective_function(**next_point)
        try:
            optimizer.register(params=next_point, target=target)
        except:
            pass
        sampling_strategy = round(next_point["sampling_strategy"],4)
        k_neighbors = next_point["k_neighbors"]
        smote_kind = next_point["smote_kind"]
        if best_target is None or target > best_target:
            best_target = target
            progress = f"|      {i}     |       {round(target,4)}   |       {sampling_strategy}     |       {k_neighbors}       |       {smote_kind}    |"
            print(simple_colors.green(progress))
        else:
            progress = f"|      {i}     |       {round(target,4)}   |       {sampling_strategy}     |       {k_neighbors}       |       {smote_kind}    |"
            print(progress)
    results = optimizer.res
    objective_values = [x['target'] for x in results]
    df = pd.read_csv("../Data/process.csv")
    if len(df) < len(objective_values):
        missing_rows = len(objective_values) - len(df)
        for _ in range(missing_rows):
            df = df.append(pd.Series(), ignore_index=True)
    df[model] = pd.Series(objective_values)
    df.to_csv("../Data/process.csv", index=False)

    best_params = optimizer.max['params']
    best_params["smote_kind"] = int(best_params["smote_kind"])
    best_params["k_neighbors"] = int(best_params["k_neighbors"])

    sampling_strategy = best_params["sampling_strategy"]
    k_neighbors = best_params["k_neighbors"]
    smote_kind = best_params["smote_kind"]
    optimized_synthetic_data, original_data = generate_data.generate_synthetic_data(sampling_strategy=sampling_strategy,
                                                                                    k_neighbors=k_neighbors,
                                                                                    smote_kind=smote_kind)
    synthetic_performance, synthetic_auc = chosen_model(optimized_synthetic_data)
    print("Model performance on the synthetic data: ", f" | AUC-score: {synthetic_auc:.4f} | Overall Performance: {synthetic_performance:.4f} |")
    original_performance, original_auc = chosen_model(original_data)
    print("Model performance on the original data: ", f" | AUC-score: {original_auc:.4f} | Overall Performance: {original_performance:.4f} |")

    name = increment_filename(path="../Data/Synthetic Data/", name=model)
    imbalance = sum(optimized_synthetic_data["TARGET"]) / len(optimized_synthetic_data["TARGET"])
    update_performance(filename=name, model=model, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, smote_kind=smote_kind, performance=synthetic_performance, auc=synthetic_auc, original_performance=original_performance, original_auc=original_auc, imbalance=imbalance)
    optimized_synthetic_data.to_csv(name, index=False)
    return name
