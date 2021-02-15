import logging
import os
import pandas as pd
import numpy as np
from src.utils import make_sure_dir_exists

logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, df_path_train: str, df_path_test: str, id_col: str, ground_truth: str):
        self.df_path_train = df_path_train
        self.df_path_test = df_path_test
        self.id_col = id_col
        self.ground_truth = ground_truth
        self.basic_preprocessing_finished = False
        self.advanced_preprocessing_finished = False
        self._union_data()

    def _union_data(self):
        df_train = pd.read_csv(self.df_path_train)
        logger.info(f'Train samples: "{len(df_train)}"')
        df_test = pd.read_csv(self.df_path_test)
        logger.info(f'Test samples: "{len(df_test)}"')
        df_test[self.ground_truth] = np.nan
        df_test = df_test[df_train.columns] # Aligning column sequence with train df
        self.data = pd.concat([df_train, df_test], axis=0)
        logger.info(f'Total samples: "{len(self.data)}"')
        logger.info('Joined train and test sets together for the preprocessing. For training and testing, they will be separated again')

    def profile(self, title: str='Dataset profile report', html_path: str=None, show_report_in_notebook: bool=False):
        """Generates a pandas-profiling report of the dataset to be displayed in a jupyter notebook.
        Optionally saves the report as an html file

        :param html_path: If provided, the pandas-profiling report will be saved to disk
        :param show_report_in_notebook: Whether or not to show report in jupyter notebook
        :return: None
        """
        if not os.path.exists(html_path):
            logger.info('Generating the profiling report')
            from pandas_profiling import ProfileReport # Locally import ProfileReport, because in the VS Code debugger, it takes AGES to import it. TODO: Put back to module beginning after successful debuggging
            profile_report = ProfileReport(self.data, title=title)
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

            # Calculate the fill value
            if fill_method == 'median':
                fill_value = self.data[col_name].median()
            elif fill_method == 'mean':
                fill_value = self.data[col_name].mean()
            elif fill_method == 'mode':
                fill_value = self.data[col_name].mode()
            else: 
                raise ValueError(f'Unknown fill_model provided: ``{fill_method}``')
            if isinstance(fill_value, pd.Series):
                fill_value = fill_value.iloc[0]
            
            # Replace the missing values with the fill value
            logger.info(f'The {fill_method} of column ``{col_name}`` equals: {fill_value}')
            self.data[col_name].fillna(fill_value, inplace=True)

    def _encode_categories(self, col_name_to_encoding):
        """Encodes categorical variables

        :param col_name_to_encoding: Dictionary describing which variable should be encoded in what way
        :return: None
        """
        for col_name, encoding in col_name_to_encoding.items():
            if isinstance(encoding, dict):
                self.data[col_name] = self.data[col_name].map(encoding)
                logger.info(f'Converted column ``{col_name}`` using the custom mapping ``{encoding}``')
            elif encoding == 'one_hot':
                dummies = pd.get_dummies(self.data[col_name], prefix=col_name)
                self.data = pd.concat([self.data, dummies], axis=1)
                logger.info(f'One-hot encoded the column ``{col_name}``')
            else:
                raise ValueError('Unknown category encoding provided: ``{encoding}``')

    def do_basic_preprocessing(self, col_name_to_fill_method: dict, col_name_to_encoding: dict):
        """Conducts basic preprocessing, i.e. replaces missing values and encodes categorical values

        :param col_name_to_fill_method: Dictionary describing which column's missing values should
            be filled and how
        :param col_name_to_encoding: Dictionary describing which column's categorical values should
            be encoded and how
        """
        if not self.basic_preprocessing_finished:
            self._fill_missing_values(col_name_to_fill_method)
            self._encode_categories(col_name_to_encoding)
            self.basic_preprocessing_finished = True
            logger.info('Preprocessing finished')
            logger.info(f'Available columns: {list(self.data.columns)}')
        else:
            logger.info('Preprocessing was already conducted')
    
    def select(self, cols_to_drop: list, mode: str='training'):
        """Returns a subset of the dataframe

        :param predictors: List of predictors to include
        :return: Dataframe subset
        """
        cols_to_keep = [col for col in self.data.columns if col not in cols_to_drop]
        if mode == 'training':
            rows_to_keep = ~self.data[self.ground_truth].isna()
            df_subset = self.data.loc[rows_to_keep, cols_to_keep]
        elif mode == 'testing':
            rows_to_keep = self.data[self.ground_truth].isna()
            df_subset = self.data.loc[rows_to_keep, cols_to_keep]
        else:
            raise ValueError(f'Unknown model provided: "{mode}"')
        return df_subset

    def do_advanced_preprocessing(
        self, 
        name_col: str='Name', 
        cabin_col: str='Cabin', 
        sibbling_spouse_col: str='SibSp', 
        parent_child_col: str='Parch',
        fare_col: str='Fare',
        age_col: str='Age'
    ):
        """More advanced preprocessing based on:

        - https://www.kaggle.com/imoore/titanic-the-only-notebook-you-need-to-see
        - https://www.kaggle.com/startupsci/titanic-data-science-solutions
        """
        if not self.advanced_preprocessing_finished:
            # Generate continuous variables
            self.data['name_len'] = self.data[name_col].apply(lambda name: len(name.split()))
            self.data['family_size'] = self.data[sibbling_spouse_col] + self.data[parent_child_col]

            # Generate indicators
            self.data['has_cabin'] = self.data[cabin_col].apply(lambda cabin: 1 if cabin is not None else 0)
            self.data['is_alone'] = self.data['family_size'].apply(lambda fam_size: 1 if fam_size == 0 else 0)
            
            # Categorize continuous variables
            self.data['fare_category'] = pd.cut(self.data[fare_col], bins=4, labels=False)
            self.data['age_category'] = pd.cut(self.data[age_col], bins=4, labels=False)
            
            # Extract the title from the name column
            self.data['title'] = self.data[name_col].str.extract(' ([A-Za-z]+)\.', expand=False) # (\w+\.) matches the first word ending with a dot character
            self.data['title'] = self.data['title'].str.lower()
                        
            # Handle missing titles
            most_common_title = self.data['title'].mode().iloc[0]
            self.data['title'].fillna(most_common_title)

            # One hot encode the titles
            dummy_titles = pd.get_dummies(self.data['title'], prefix='title')
            self.data = pd.concat([self.data, dummy_titles], axis=1)
            
            # Logging and keeping track of state variables
            logger.info(f'Available columns: {list(self.data.columns)}')
            self.advanced_preprocessing_finished = True
        else:
            logger.info('Advanced preprocessing was already done')