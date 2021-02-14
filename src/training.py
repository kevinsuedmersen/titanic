import logging
import pickle
import pandas as pd
from src.utils import make_sure_dir_exists
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.options.display.width = 0
logger = logging.getLogger(__name__)
MODEL_DICT = {'svm': SVC}


class Model:
    def __init__(self, model_name, model_path, ground_truth, id_col_name, scaling_mode, scaler_path, **kwargs):
        self.model_name = model_name
        self.model_path = model_path
        self.ground_truth = ground_truth
        self.id_col_name = id_col_name
        self.scaler_path = scaler_path
        self.scaling_mode = scaling_mode
        self.scaler = None
        model_cls = MODEL_DICT[model_name]
        self.model = model_cls(**kwargs)
        self.trained = False

    def _get_inputs(self, df):
        # Get a subset of the predictors
        predictors = df.columns.difference([self.ground_truth, self.id_col_name])
        X_raw = df[predictors].values

        # Scale the predictors
        if self.scaling_mode == 'min_max':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f'Unknown scaling_mode provided: {self.scaling_mode}')
        self.scaler.fit(X_raw)
        X_scaled = self.scaler.transform(X_raw)
        
        # Save scaler for generating the submissions later
        self._pickle(self.scaler, self.scaler_path)

        return X_scaled
    
    def _get_train_val_data(self, df, test_size, random_state):
        X = self._get_inputs(df)
        y = df[self.ground_truth].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_val, y_train, y_val

    @staticmethod
    def _pickle(obj, path):
         with open(path, 'wb') as file:
            pickle.dump(obj, file)
            logger.info(f'Saved the trained model to ``{path}``')

    def train_and_evaluate(self, df, val_size=0.3, random_state=42):
        # Train the model
        logger.info(f'Training ``{self.model_name}`` started')
        X_train, X_val, ytrue_train, ytrue_val = self._get_train_val_data(df, val_size, random_state)
        self.model.fit(X_train, ytrue_train)
        logger.info('Training finished')

        # Evaluate it on the validation set
        ypred_val = self.model.predict(X_val)
        acc_val = metrics.accuracy_score(ytrue_val, ypred_val)
        prec_val = metrics.precision_score(ytrue_val, ypred_val)
        rec_val = metrics.recall_score(ytrue_val, ypred_val)
        f1_val = metrics.f1_score(ytrue_val, ypred_val)
        logger.info(f'Results on the validation set: accuracy {acc_val}, precision {prec_val}, recall {rec_val}, f1 {f1_val}')

        # Save the model for later use
        self._pickle(self.model, self.model_path)
        self.trained = True

    def hparam_tuning(self, grid_search_dict):
        # TODO: HP tuning

        # Save model for later use
        self._pickle()
    
    def gen_submission_file(self, df_test, submission_path):
        assert self.trained, 'Model must be trained, before the submission file can be generated'

        # Generate predictions on the test set
        X_test = self._get_inputs(df_test)
        logger.info('Generating predictions on the test set')
        ypred_test = self.model.predict(X_test)
        
        # Put the predictions into the submission file format and save it
        ids = df_test[self.id_col_name]
        submission_dict = {self.id_col_name: ids, self.ground_truth: ypred_test}
        submission_df = pd.DataFrame(submission_dict)
        make_sure_dir_exists(submission_path)
        submission_df.to_csv(submission_path, index=False)
        logger.info(f'Saved submission to ``{submission_path}``')