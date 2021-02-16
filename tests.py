from src.pipeline import MLPipeline

# Setup
MISSING_VALUE_CONFIG = {
    'Age': 'median', 
    'Embarked': 'mode',
    'Fare': 'median'
}
ENCODING_CONFIG = {
    'Pclass': {1: 3, 2: 2, 3: 1}, # original_value: encoding_value
    'Sex': {'male': 1, 'female': 0},
    'Embarked': 'one_hot'
}

MODEL_CONFIG = {
    'svm': {'kernel': 'rbf'},
    'decision_tree': {},
    'random_forest': {}
}
ml_pipeline = MLPipeline(
    df_path_train='data/train.csv', 
    df_path_test='data/test.csv', 
    id_col='PassengerId', 
    ground_truth='Survived', 
    model_config=MODEL_CONFIG
)

# EDA
#ml_pipeline.run_eda()

# Iteration 1
ml_pipeline.run(
    missing_value_config=MISSING_VALUE_CONFIG, 
    encoding_config=ENCODING_CONFIG, 
    advanced_preprocessing=False
)

# Iteration 2
ml_pipeline.run(
    missing_value_config=MISSING_VALUE_CONFIG, 
    encoding_config=ENCODING_CONFIG, 
    advanced_preprocessing=True
)

# Iteration 3