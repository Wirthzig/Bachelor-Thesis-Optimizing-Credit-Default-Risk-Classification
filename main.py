import submission
import bayesian_optimization
import warnings

warnings.filterwarnings('ignore')

################################################
train_file = "../Data/GiveMeSomeCredit/cs-training-preprocessed.csv"
test_file = "../Data/GiveMeSomeCredit/cs-testing-preprocessed.csv"
init_points = 5
n_iter = 25
test_size = 0.25
################################################

model = "Logistic Regression"
# Bayesian Optimization
optimized_synthetic_data_path = bayesian_optimization.optimize(model=model, init_points=init_points, n_iter=n_iter)
submission.create_submission(optimized_synthetic_data_path, test_file=test_file, applied_model=model)
# Random Search
optimized_synthetic_data_path = bayesian_optimization.optimize_Random_Search(model=model, init_points=init_points, n_iter=n_iter)
submission.create_submission(optimized_synthetic_data_path, test_file=test_file, applied_model=model)
# Original
submission.create_submission(train_file, test_file=test_file, applied_model=model, original=True)

model = "LightGBM"
#Bayesian Optimization
optimized_synthetic_data_path = bayesian_optimization.optimize(model=model, init_points=init_points, n_iter=n_iter)
submission.create_submission(optimized_synthetic_data_path, test_file=test_file, applied_model=model)
# Random Search
optimized_synthetic_data_path = bayesian_optimization.optimize_Random_Search(model=model, init_points=init_points, n_iter=n_iter)
submission.create_submission(optimized_synthetic_data_path, test_file=test_file, applied_model=model)
# Original
submission.create_submission(train_file, test_file=test_file, applied_model=model, original=True)

model = "Neural Network"
# Bayesian Optimization
optimized_synthetic_data_path = bayesian_optimization.optimize(model=model, init_points=init_points, n_iter=n_iter)
submission.create_submission(optimized_synthetic_data_path, test_file=test_file, applied_model=model)
# Random Search
optimized_synthetic_data_path = bayesian_optimization.optimize_Random_Search(model=model, init_points=init_points, n_iter=n_iter)
submission.create_submission(optimized_synthetic_data_path, test_file=test_file, applied_model=model)
# Original
submission.create_submission(train_file, test_file=test_file, applied_model=model, original=True)
