import logging
import os
import pandas as pd
from src.utils import make_sure_dir_exists

logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, df_path: str, ground_truth: str=None, id_col: str=None):
        self.df_path = df_path
        self.ground_truth = ground_truth
        self.id_col = id_col
        self.df = pd.read_csv(df_path)
        logger.info(F'Read dataframe from ``{df_path}`` into memory')
        self.preprocessing_finished = False

    def profile(self, title: str='Dataset profile report', html_path: str=None, show_report_in_notebook: bool=False):
        """Generates a pandas-profiling report of the dataset to be displayed in a jupyter notebook.
        Optionally saves the report as an html file

        :param html_path: If provided, the pandas-profiling report will be saved to disk
        :param show_report_in_notebook: Whether or not to show report in jupyter notebook
        :return: None
        """
        if not os.path.exists(html_path):
            from pandas_profiling import ProfileReport # Locally import ProfileReport, because in the VS Code debugger, it takes AGES to import it. TODO: Put back to module beginning after successful debuggging
            logger.info('Generating the profiling report')
            profile_report = ProfileReport(self.df, title=title)
            if html_path is not None:
                make_sure_dir_exists(html_path)
                profile_report.to_file(html_path)
                logger.info(f'Saved the pandas-profiling report to ``{html_path}``')
                if show_report_in_notebook:
                    profile_report.to_notebook_iframe()
        else:
            logger.info(f'A profiling report was already generated and is located at ``{html_path}``')

    def _fill_missing_values(self, col_name_to_fill_method: dict):
        """Fills missing values

        :param col_name_to_fill_method: Dictionary mapping column name to method by which missing values
            are filled
        :return: None
        """
        for col_name, fill_method in col_name_to_fill_method.items():

            # Calculate the fill mode
            if fill_method == 'median':
                fill_value = self.df[col_name].median()
            elif fill_method == 'mean':
                fill_value = self.df[col_name].mean()
            elif fill_method == 'mode':
                fill_value = self.df[col_name].mode()
            else: 
                raise ValueError(f'Unknown fill_model provided: ``{fill_method}``')
            if isinstance(fill_value, pd.Series):
                fill_value = fill_value.iloc[0]
            
            # Replace the missing values with the fill value
            logger.info(f'The {fill_method} of column ``{col_name}`` equals: {fill_value}')
            self.df[col_name].fillna(fill_value, inplace=True)

    def _encode_categories(self, col_name_to_encoding):
        """Encodes categorical variables

        :param col_name_to_encoding: Dictionary describing which variable should be encoded in what way
        :return: None
        """
        for col_name, encoding in col_name_to_encoding.items():
            if isinstance(encoding, dict):
                self.df[col_name] = self.df[col_name].apply(lambda cat_name: encoding[cat_name])
                logger.info(f'Converted column ``{col_name}`` using the custom mapping ``{encoding}``')
            elif encoding == 'one_hot':
                dummies = pd.get_dummies(self.df[col_name], prefix=col_name)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col_name, axis=1, inplace=True)
                logger.info(f'One-hot encoded the column ``{col_name}``')
            else:
                raise ValueError('Unknown category encoding provided: ``{encoding}``')

    def preprocess_df_iteration_1(self, col_name_to_fill_method: dict, col_name_to_encoding: dict, predictors: list):
        """Conducts the complete preprocessing pipeline for iteration 1

        :param col_name_to_fill_method: Dictionary describing which column's missing values should
            be filled and how
        :param col_name_to_encoding: Dictionary describing which column's categorical values should
            be encoded and how
        :param predictors: List of predictors to include
        """
        if not self.preprocessing_finished:
            # Select a subset of predictors and always select the ground truth variable
            if self.ground_truth is not None:
                cols_to_keep = predictors + [self.ground_truth, self.id_col]
            else:
                cols_to_keep = predictors + [self.id_col]
            self.df = self.df[cols_to_keep]
            
            # Fill Missing values
            self._fill_missing_values(col_name_to_fill_method)

            # Custom category encoding
            self._encode_categories(col_name_to_encoding)

            # Set preprocessing_finished to True for idempotency
            self.preprocessing_finished = True
            logger.info('Preprocessing finished')
        else:
            logger.info('Preprocessing was already conducted')
