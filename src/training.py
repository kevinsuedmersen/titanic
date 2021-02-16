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
        """Instantiates the model class

        :param model_name: Model name
        :type model_name: str
        :param ground_truth: Name of ground truth col
        :type ground_truth: str
        :param id_col_name: Name of id vol
        :type id_col_name: str
        :param scaling_mode: Scaling method applied to input features, defaults to None
        :type scaling_mode: str, optional
        :param results_dir: Directory where results will be stored, defaults to 'results'
        :type results_dir: str, optional
        """
        self.model_name = model_name
        self.model_path = os.path.join(results_dir, model_name + '.pickle')
        self.ground_truth = ground_truth
        self.id_col_name = id_col_name
        self.scaling_mode = scaling_mode
        self.results_dir = results_dir
        
        # Instantiate the model and other placeholders
        model_cls = MODEL_DICT[model_name]
        self.model = model_cls(**kwargs)
        self.trained = False
        self.scaler = None

    def _get_inputs(self, df: pd.DataFrame):
        """Selects the input features, converts them to numpy arrays and optionally scales them

        :param df: Dataframe with all preprocessed data
        :type df: pd.DataFrame
        :return: (Scaled) input features
        :rtype: np.array
        """
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
            scaler_path = os.path.join(self.results_dir, self.scaling_mode + '.pickle')
            self._pickle(self.scaler, scaler_path)

        return X
    
    def _get_train_val_data(self, df: pd.DataFrame, test_size: float, random_state=42):
        """Splits the training data into training and validation subset

        :param df: Dataframe with all preprocessed data
        :type df: pd.DataFrame
        :param test_size: Fraction of data reserved for validating
        :type test_size: flow
        :param random_state: Random state when splitting dataset, defaults to 42
        :type random_state: int, optional
        :return: X, the potentially scaled input features and y the outputs
        :rtype: tuple
        """
        X = self._get_inputs(df)
        y = df[self.ground_truth].values
        if test_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X_train, X_val, y_train, y_val
        else:
            return X, y

    @staticmethod
    def _pickle(obj: object, path: str):
        """Serializes obj to path

        :param obj: Some python object
        :type obj: object
        :param path: Path where to save obj
        :type path: str
        """
         with open(path, 'wb') as file:
            pickle.dump(obj, file)
            logger.info(f'Saved the trained model to ``{path}``')

    def train_and_evaluate(self, train_df: pd.DataFrame, val_size: float=0.3, random_state: int=42):
        """Trains the model and evaluates it on the validation set

        :param train_df: Dataframe with the training data
        :type train_df: pd.DataFrame
        :param val_size: Fraction of data reserved for validating, defaults to 0.3
        :type val_size: float, optional
        :param random_state: Random state, defaults to 42
        :type random_state: int, optional
        """
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

    def hparam_tuning(self, train_df: pd.DataFrame, grid_search_dict: dict):
        """Conducts a grid search to find the optimal hyper params

        :param train_df: Training dataframe
        :type train_df: Dataframe with the training data
        :param grid_search_dict: Dict containing a mapping between hyper-param and its values to test
        :type grid_search_dict: dict
        """
        gs = GridSearchCV(self.model, grid_search_dict, cv=3, scoring='accuracy', verbose=1, refit=True, n_jobs=-2)
        X_train, y_train = self._get_train_val_data(train_df, test_size=0)
        logger.info('Grid search started')
        gs.fit(X_train, y_train)
        logger.info('Grid search ended')

        # Save model for later use
        self.model = gs.best_estimator_
        self._pickle(self.model, self.model_path)
    
    def gen_submission_file(self, df_test: pd.DataFrame, submission_name: str):
        """Generates the submission file

        :param df_test: Dataframe with the test data
        :type df_test: pd.DataFrame
        :param submission_name: Name of this submission which will be appended to the submission file
        :type submission_name: str
        """
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