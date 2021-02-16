import logging
import os
import pickle
import pandas as pd
from src.utils import make_sure_dir_exists
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

pd.options.display.width = 0
logger = logging.getLogger(__name__)
MODEL_DICT = {
    'svm': SVC, 
    'decision_tree': DecisionTreeClassifier, 
    'random_forest': RandomForestClassifier
}
SCALER_DICT = {
    'min_max': MinMaxScaler
}


class Model:
    def __init__(
        self, 
        model_name: str, 
        ground_truth: str, 
        id_col_name: str, 
        scaling_mode: str=None, 
        results_dir: str='results', 
        **kwargs
    ):
        self.model_name = model_name
        self.model_path = os.path.join(results_dir, model_name + '.pickle')
        self.ground_truth = ground_truth
        self.id_col_name = id_col_name
        self.scaler_path = os.path.join(results_dir, scaling_mode + '.pickle')
        self.scaling_mode = scaling_mode
        self.results_dir = results_dir
        
        # Instantiate the model and other placeholders
        model_cls = MODEL_DICT[model_name]
        self.model = model_cls(**kwargs)
        self.trained = False
        self.scaler = None

    def _get_inputs(self, df):
        # Get a subset of the predictors
        predictors = df.columns.difference([self.ground_truth, self.id_col_name])
        X = df[predictors].values

        # Scale the predictors
        if self.scaling_mode is not None:
            if self.scaler is None:
                scaler_cls = SCALER_DICT[self.scaling_mode]
                self.scaler = scaler_cls()
                self.scaler.fit(X)
                X = self.scaler.transform(X)
            else:
                X = self.scaler.transform(X)
            self._pickle(self.scaler, self.scaler_path)

        return X
    
    def _get_train_val_data(self, df, test_size, random_state=42):
        X = self._get_inputs(df)
        y = df[self.ground_truth].values
        if test_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X_train, X_val, y_train, y_val
        else:
            return X, y

    @staticmethod
    def _pickle(obj, path):
         with open(path, 'wb') as file:
            pickle.dump(obj, file)
            logger.info(f'Saved the trained model to ``{path}``')

    def train_and_evaluate(self, train_df, val_size=0.3, random_state=42):
        # Train the model
        logger.info(f'Training ``{self.model_name}`` started')
        X_train, X_val, ytrue_train, ytrue_val = self._get_train_val_data(train_df, val_size, random_state)
        self.model.fit(X_train, ytrue_train)
        logger.info('Training finished')

        # Evaluate it on the validation set
        ypred_val = self.model.predict(X_val)
        acc_val = metrics.accuracy_score(ytrue_val, ypred_val)
        prec_val = metrics.precision_score(ytrue_val, ypred_val)
        rec_val = metrics.recall_score(ytrue_val, ypred_val)
        f1_val = metrics.f1_score(ytrue_val, ypred_val)
        logger.info(f'Results on the validation set: accuracy {acc_val:.2f}, precision {prec_val:.2f}, recall {rec_val:.2f}, f1 {f1_val:.2f}')

        # Save the model for later use
        self._pickle(self.model, self.model_path)
        self.trained = True

    def hparam_tuning(self, train_df, grid_search_dict):
        gs = GridSearchCV(self.model, grid_search_dict, cv=3, scoring='accuracy', verbose=1, refit=True, n_jobs=-2)
        X_train, y_train = self._get_train_val_data(train_df, test_size=0)
        logger.info('Grid search started')
        gs.fit(X_train, y_train)
        logger.info('Grid search ended')

        # Save model for later use
        self.model = gs.best_estimator_
        self._pickle(self.model, self.model_path)
    
    def gen_submission_file(self, df_test, submission_name):
        assert self.trained, 'Model must be trained, before the submission file can be generated'

        # Generate predictions on the test set
        X_test = self._get_inputs(df_test)
        logger.info('Generating predictions on the test set')
        ypred_test = self.model.predict(X_test).astype('int')
        
        # Put the predictions into the submission file format and save it
        ids = df_test[self.id_col_name]
        submission_dict = {self.id_col_name: ids, self.ground_truth: ypred_test}
        submission_df = pd.DataFrame(submission_dict)
        submission_path = os.path.join(self.results_dir, f'{self.model_name}_{submission_name}.csv')
        make_sure_dir_exists(submission_path)
        submission_df.to_csv(submission_path, index=False)
        logger.info(f'Saved submission to ``{submission_path}``')