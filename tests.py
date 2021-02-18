"""This file is just for debugging and may be disregarded. It contains the same code like results.ipynb
"""

from src.pipeline import MLPipeline


if __name__ == '__main__':
    # EDA
    ml_pipeline = MLPipeline(
        df_path_train='resources/data/train.csv', 
        df_path_test='resources/data/test.csv', 
        id_col='PassengerId', 
        ground_truth='Survived',
        results_dir='results'
    )
    #ml_pipeline.run_eda()

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
        'svm': {'kernel': 'rbf', 'scaling_mode': 'min_max'},
        'decision_tree': {'scaling_mode': None},
        'random_forest': {'scaling_mode': 'min_max'},
    }

    # Iteration 1
    ml_pipeline.run(
        missing_value_config=MISSING_VALUE_CONFIG, 
        encoding_config=ENCODING_CONFIG, 
        advanced_preprocessing=False,
        model_config=MODEL_CONFIG
    )

    # Iteration 2
    ml_pipeline.run(
        missing_value_config=MISSING_VALUE_CONFIG, 
        encoding_config=ENCODING_CONFIG, 
        advanced_preprocessing=True,
        model_config=MODEL_CONFIG
    )

    # Iteration 3
    MODEL_CONFIG = {
        'svm': {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'], 
            'scaling_mode': 'min_max'
        },
        'decision_tree': {
            'min_samples_leaf': [1, 10, 100],
            'criterion': ['gini', 'entropy'],
            'min_impurity_decrease': [0.0, 0.1], 
            'scaling_mode': None
        },
        'random_forest': {
            'n_estimators': [100, 1000],
            'criterion': ['gini', 'entropy'],
            'scaling_mode': 'min_max'
        },
    }
    ml_pipeline.run(
        missing_value_config=MISSING_VALUE_CONFIG, 
        encoding_config=ENCODING_CONFIG, 
        advanced_preprocessing=True,
        model_config=MODEL_CONFIG,
        hp_tuning=True
    )