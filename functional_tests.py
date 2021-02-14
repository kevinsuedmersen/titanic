from src.training import Model
from src.preprocessing import Dataset
from src.utils import set_root_logger


# if __name__ == 'main' doesn't seem to be working in the VS code debugger
set_root_logger()

# Exploratory data analysis
train_ds = Dataset(df_path='/home/kevinsuedmersen/dev/titanic/data/train.csv')
train_ds.profile(title='train_ds_profile', html_path='results/train_ds_profiling.html')
test_ds = Dataset(df_path='/home/kevinsuedmersen/dev/titanic/data/test.csv')
test_ds.profile(title='test_ds_profile', html_path='results/test_ds_profiling.html')

# Basic preprocessing configuration
predictors = [
    'Pclass', 
    'Age', 
    'SibSp', 
    'Parch', 
    'Fare', 
    'Sex_female', 
    'Sex_male', 
    'Embarked_C', 
    'Embarked_Q', 
    'Embarked_S'
]
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

# Preprocess the training data and get a subset of predictors
train_ds.clean(col_name_to_fill_method, col_name_to_encoding)
train_df = train_ds.get_df_subset(predictors, id_col='PassengerId', ground_truth='Survived')

# Preprocess the test data and get a subset of predictors
test_ds.clean(col_name_to_fill_method, col_name_to_encoding)
test_df = test_ds.get_df_subset(predictors, id_col='PassengerId')

# Train the model and generate the submission file
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
model.gen_submission_file(test_df, submission_path='results/submission.csv')
