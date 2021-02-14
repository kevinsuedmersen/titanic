from src.training import Model
from src.preprocessing import Dataset
from src.utils import set_root_logger


# if __name__ == 'main' doesn't seem to be working in the VS code debugger
set_root_logger()

# Exploratory data analysis
train_ds = Dataset(
    df_path='/home/kevinsuedmersen/dev/titanic/data/train.csv',
    ground_truth='Survived',
    id_col='PassengerId'
)
train_ds.profile(title='train_ds_profile', html_path='results/train_ds_profiling.html')
test_ds = Dataset(
    df_path='/home/kevinsuedmersen/dev/titanic/data/test.csv',
    ground_truth=None,
    id_col='PassengerId'
)
test_ds.profile(title='test_ds_profile', html_path='results/test_ds_profiling.html')

# Preprocessing
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
col_name_to_fill_method = {
    'Age': 'median', 
    'Embarked': 'mode',
    'Fare': 'median'
}
col_name_to_encoding = {
    'Pclass': {1: 3, 2: 2, 3: 1}, # original_value: encoding_value
    'Sex': 'one_hot',
    'Embarked': 'one_hot'
}
train_ds.preprocess(col_name_to_fill_method, col_name_to_encoding, predictors)
test_ds.preprocess(col_name_to_fill_method, col_name_to_encoding, predictors)

# Simple training run
model = Model(
    model_name='svm',
    model_path='results/svm_model.pickle', 
    ground_truth='Survived', 
    id_col_name='PassengerId',
    scaling_mode='min_max',
    scaler_path='results/min_max_scaler.pickle',
    kernel='rbf'
)
model.train_and_evaluate(train_ds.df)
model.gen_submission_file(test_ds.df, submission_path='results/submission.csv')
