import logging
import os
import pandas as pd
import numpy as np
from src.utils import clean_ticket, get_leading_ticket_number_digit, get_ticket_group, get_ticket_number, get_ticket_number_digit_len, get_ticket_prefix, make_sure_dir_exists
from IPython.display import display, IFrame

logger = logging.getLogger(__name__)
pd.options.display.width = 0

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
            # TODO: Put import to the right place
            from pandas_profiling import ProfileReport 
            profile_report = ProfileReport(self.data, title=title)
            if html_path is not None:
                make_sure_dir_exists(html_path)
                profile_report.to_file(html_path)
                logger.info(f'Saved the pandas-profiling report to ``{html_path}``')
            if show_report_in_notebook:
                profile_report.to_notebook_iframe()
        else:
            logger.info(f'A profiling report was already generated and will be loaded from ``{html_path}``')
            display(IFrame(src=html_path, width=10**3, height=10**3))

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
    
    def drop_cols(self, cols_to_drop: list, mode: str='training'):
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

    def _gen_continuous_vars(self, name_col, sibbling_spouse_col, parent_child_col):
        # Generate continuous variables
        self.data['name_len'] = self.data[name_col].apply(lambda name: len(name.split()))
        self.data['family_size'] = self.data[sibbling_spouse_col] + self.data[parent_child_col]
    
    def _gen_indicator_vars(self, cabin_col):
        # Generate indicators
        self.data['has_cabin'] = self.data[cabin_col].apply(lambda cabin: 1 if cabin is not None else 0)
        self.data['is_alone'] = self.data['family_size'].apply(lambda fam_size: 1 if fam_size == 0 else 0)

    def _bin_continuous_vars(self, fare_col, age_col, n_bins: int=4):
        # Bin continuous variables
        self.data['fare_category'] = pd.cut(self.data[fare_col], bins=n_bins, labels=False)
        self.data['age_category'] = pd.cut(self.data[age_col], bins=n_bins, labels=False)

    def _extract_title_from_name(self, name_col):
        # Extract the title from the name column
        self.data['title'] = self.data[name_col].str.extract(' ([A-Za-z]+)\.', expand=False) # (\w+\.) matches the first word ending with a dot character
        self.data['title'] = self.data['title'].str.lower()
                    
        # Handle missing titles
        most_common_title = self.data['title'].mode().iloc[0]
        self.data['title'].fillna(most_common_title)

        # One hot encode the titles
        dummy_titles = pd.get_dummies(self.data['title'], prefix='title')
        self.data = pd.concat([self.data, dummy_titles], axis=1)

    def _extract_ticket_info(self, ticket_col):
        """Extracts information from the ticket column. Mostly based on: 

        - https://www.kaggle.com/pliptor/titanic-ticket-only-study
        :return: None
        """
        # For each ticket, count its number of duplicates
        n_duplicate_tickets = self.data.groupby(ticket_col).size()
        self.data['n_duplicate_tickets'] = self.data[ticket_col].map(n_duplicate_tickets)
        
        # Make the description of crew member tickets consistent with the following feature extraction
        self.data[ticket_col] = self.data[ticket_col].replace('LINE','LINE 0')

        # Clean the ticket column
        self.data[ticket_col] = self.data[ticket_col].apply(lambda ticket: clean_ticket(ticket))

        # Extracting the ticket prefix
        self.data['ticket_prefix'] = self.data[ticket_col].apply(lambda ticket: get_ticket_prefix(ticket))

        # Extract The ticket number
        self.data['ticket_number'] = self.data[ticket_col].apply(lambda ticket: get_ticket_number(ticket))
        
        # Extract the digit length from the ticket
        self.data['ticket_number_digit_len'] = self.data['ticket_number'] \
            .apply(lambda ticket_number : get_ticket_number_digit_len(ticket_number))
        
        # Extract the leading digit from the ticket number
        self.data['leading_digit'] = self.data['ticket_number'] \
            .apply(lambda ticket_number: get_leading_ticket_number_digit(ticket_number))
        
        # Extract the ticket groups from the ticket number
        self.data['ticket_group'] = self.data['ticket_number'] \
            .apply(lambda ticket_number: get_ticket_group(ticket_number))

        # One hot encode the ticket prefixes and ticket groups
        ticket_prefix_dummies = pd.get_dummies(self.data['ticket_prefix'], prefix='ticket_prefix')
        ticket_group_dummies = pd.get_dummies(self.data['ticket_group'], prefix='ticket_group')
        self.data = pd.concat([self.data, ticket_prefix_dummies, ticket_group_dummies], axis=1)

    def do_advanced_preprocessing(
        self, 
        name_col: str='Name', 
        cabin_col: str='Cabin', 
        sibbling_spouse_col: str='SibSp', 
        parent_child_col: str='Parch',
        fare_col: str='Fare',
        age_col: str='Age',
        ticket_col: str='Ticket'
    ):
        """More advanced preprocessing based on:

        - https://www.kaggle.com/imoore/titanic-the-only-notebook-you-need-to-see
        - https://www.kaggle.com/startupsci/titanic-data-science-solutions
        """
        if not self.advanced_preprocessing_finished:
            self._gen_continuous_vars(name_col, sibbling_spouse_col, parent_child_col)
            self._gen_indicator_vars(cabin_col)
            self._bin_continuous_vars(fare_col, age_col)
            self._extract_title_from_name(name_col)
            self._extract_ticket_info(ticket_col)
            
            # Logging and keeping track of state variables
            logger.info(f'Number of columns: {len(self.data.columns)}')
            self.advanced_preprocessing_finished = True
        else:
            logger.info('Advanced preprocessing was already done')