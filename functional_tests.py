from src.preprocessing import Dataset
from src.utils import set_root_logger


# if __name__ == 'main' doesn't seem to be working in the VS code debugger
set_root_logger()
train_ds = Dataset('/home/kevinsuedmersen/dev/titanic/data/train.csv', ground_truth='Survived')

# Preprocessing configuration
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
col_name_to_fill_method = {
    'Age': 'median', 
    'Embarked': 'mode'
}
col_name_to_encoding = {
    'Pclass': {1: 3, 2: 2, 3: 1}, # original_value: encoding_value
    'Sex': 'one_hot',
    'Embarked': 'one_hot'
}
train_ds.preprocess_df_iteration_1(col_name_to_fill_method, col_name_to_encoding, predictors)
