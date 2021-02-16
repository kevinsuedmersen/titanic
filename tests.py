import numpy as np
from src.training import Model
from src.preprocessing import Dataset
from src.utils import set_root_logger


# if __name__ == 'main' doesn't seem to be working in the VS code debugger
set_root_logger()

# Exploratory data analysis
ds = Dataset(
    df_path_train='/home/kevinsuedmersen/dev/titanic/data/train.csv',
    df_path_test='/home/kevinsuedmersen/dev/titanic/data/test.csv',
    id_col='PassengerId',
    ground_truth='Survived'
)
# TODO: Uncomment following line in Notebook
#ds.profile(title='ds_profile_report', html_path='results/ds_profile_report.html')

####################################################################################################
# Iteration 1
####################################################################################################
# Configurations
col_name_to_fill_method = {
    'Age': 'median', 
    'Embarked': 'mode',
    'Fare': 'median'
}
col_name_to_encoding = {
    'Pclass': {1: 3, 2: 2, 3: 1}, # original_value: encoding_value
    'Sex': {'male': 1, 'female': 0},
    'Embarked': 'one_hot'
}

# Preprocessing
ds.do_basic_preprocessing(col_name_to_fill_method, col_name_to_encoding)
cols_to_drop = ['Name', 'Ticket', 'Cabin', 'Embarked']
train_df = ds.select(cols_to_drop, mode='training')
test_df = ds.select(cols_to_drop, mode='testing')

# Training, predicting and evaluating
model = Model(
    model_name='svm',
    model_path='results/svm_model.pickle', 
    ground_truth='Survived', 
    id_col_name='PassengerId',
    scaling_mode='min_max',
    scaler_path='results/min_max_scaler.pickle',
    kernel='rbf'
)
model.train_and_evaluate(train_df)
model.gen_submission_file(test_df, submission_path='results/basic_preprocessing_submission.csv')

####################################################################################################
# Iteration 2
####################################################################################################
# Preprocessing
ds.do_advanced_preprocessing()
cols_to_drop += ['title']
train_df = ds.select(cols_to_drop, mode='training')
test_df = ds.select(cols_to_drop, mode='testing')

# Training, predicting and evaluating
model = Model(
    model_name='svm',
    model_path='results/svm_model.pickle', 
    ground_truth='Survived', 
    id_col_name='PassengerId',
    scaling_mode='min_max',
    scaler_path='results/min_max_scaler.pickle',
    kernel='rbf'
)
model.train_and_evaluate(train_df)
model.gen_submission_file(test_df, submission_path='results/advanced_preprocessing_submission.csv')

####################################################################################################
# Iteration 3
####################################################################################################
# Setting hyper-parameter values
grid_search_grid = {
    'C': np.logspace(-3, 10),
    'gamma': np.logspace(-3, 10)
}
model.hparam_tuning(train_df, grid_search_grid)
model.gen_submission_file(test_df, submission_path='results/grid_search.csv')