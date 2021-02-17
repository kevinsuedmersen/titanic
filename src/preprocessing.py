import logging
import os
import pandas as pd
import numpy as np
from src.utils import clean_ticket, get_leading_ticket_number_digit, get_ticket_group, get_ticket_number, get_ticket_number_digit_len, get_ticket_prefix
from IPython.display import display, IFrame
from pandas_profiling import ProfileReport

logger = logging.getLogger(__name__)
pd.options.display.width = 0

class Dataset:
    def __init__(self, df_path_train: str, df_path_test: str, id_col: str, ground_truth: str):
        """Instantiates a dataset and concatenates the training and test set for joint preprocessing

        :param df_path_train: Path to training data
        :type df_path_train: str
        :param df_path_test: Path to test data
        :type df_path_test: str
        :param id_col: Id column
        :type id_col: str
        :param ground_truth: Ground truth column
        :type ground_truth: str
        """
        self.df_path_train = df_path_train
        self.df_path_test = df_path_test
        self.id_col = id_col
        self.ground_truth = ground_truth
        self.basic_preprocessing_finished = False
        self.advanced_preprocessing_finished = False
        self._union_data()

    def _union_data(self):
        """Vertically concatenates the training and test data for joint preprocessing
        """
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
            profile_report = ProfileReport(self.data, title=title)
            if html_path is not None:
                profile_report.to_file(html_path)
                logger.info(f'Saved the pandas-profiling report to ``{html_path}``')
            profile_report.to_notebook_iframe()
        else:
            logger.info(f'A profiling report was already generated and will be loaded from ``{html_path}``')
            display(IFrame(src=html_path, width=10**3, height=10**3))

    def _fill_missing_values(self, missing_value_config: dict):
        """Fills missing values as specified in missing_value_config

        :param missing_value_config: Dictionary mapping column name to missing value filling method
        :type missing_value_config: dict
        :raises ValueError: Raises ValueError when unknown filling method is provided
        """
        for col_name, fill_method in missing_value_config.items():

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

    def _encode_categories(self, encoding_config: dict):
        """Encodes categorical variables according to encoding_config

        :param encoding_config: Dict mapping column name to encoding method
        :type encoding_config: dict
        :raises ValueError: If unknown encoding method is provided
        """
        for col_name, encoding in encoding_config.items():
            if isinstance(encoding, dict):
                self.data[col_name] = self.data[col_name].map(encoding)
                logger.info(f'Converted column ``{col_name}`` using the custom mapping ``{encoding}``')
            elif encoding == 'one_hot':
                dummies = pd.get_dummies(self.data[col_name], prefix=col_name)
                self.data = pd.concat([self.data, dummies], axis=1)
                logger.info(f'One-hot encoded the column ``{col_name}``')
            else:
                raise ValueError('Unknown category encoding provided: ``{encoding}``')

    def do_basic_preprocessing(self, missing_value_config: dict, encoding_config: dict):
        """Conducts basic preprocessing, i.e. fills missing values and encodes existing, categorical 
        variables

        :param missing_value_config: Dict mapping column name to filling method
        :type missing_value_config: dict
        :param encoding_config: Dict mapping colname to encoding method
        :type encoding_config: dict
        """
        if not self.basic_preprocessing_finished:
            self._fill_missing_values(missing_value_config)
            self._encode_categories(encoding_config)
            self.basic_preprocessing_finished = True
            logger.info('Preprocessing finished')
            logger.info(f'Available columns: {list(self.data.columns)}')
        else:
            logger.info('Preprocessing was already conducted')
    
    def drop_cols(self, cols_to_drop: list, mode: str='training'):
        """Drops a subset of cols

        :param cols_to_drop: Cols to drop
        :type cols_to_drop: list
        :param mode: Whether we are training or testing/predicting, defaults to 'training'
        :type mode: str, optional
        :raises ValueError: If unknown mode is provided
        :return: The dataframe subset
        :rtype: pd.DataFrame
        """
        cols_to_keep = [col for col in self.data.columns if col not in cols_to_drop]
        if mode == 'training':
            rows_to_keep = ~self.data[self.ground_truth].isna()
            df_subset = self.data.loc[rows_to_keep, cols_to_keep]
        elif mode == 'testing':
            rows_to_keep = self.data[self.ground_truth].isna()
            df_subset = self.data.loc[rows_to_keep, cols_to_keep]
        else:
            raise ValueError(f'Unknown mode provided: "{mode}"')
        return df_subset

    def _gen_continuous_vars(self, name_col: str, sibbling_spouse_col: str, parent_child_col: str):
        """Generates continuous variables

        :param name_col: Name of name col
        :type name_col: str
        :param sibbling_spouse_col: Name of SibSp col
        :type sibbling_spouse_col: str
        :param parent_child_col: Name of Parch col
        :type parent_child_col: str
        """
        # Generate continuous variables
        self.data['name_len'] = self.data[name_col].apply(lambda name: len(name.split()))
        self.data['family_size'] = self.data[sibbling_spouse_col] + self.data[parent_child_col]
    
    def _gen_indicator_vars(self, cabin_col: str):
        """Generates indicator variables

        :param cabin_col: Cabin col
        :type cabin_col: str
        """
        # Generate indicators
        self.data['has_cabin'] = self.data[cabin_col].apply(lambda cabin: 1 if cabin is not None else 0)
        self.data['is_alone'] = self.data['family_size'].apply(lambda fam_size: 1 if fam_size == 0 else 0)

    def _bin_continuous_vars(self, fare_col: str, age_col: str, n_bins: int=4):
        """Bins continuous vars

        :param fare_col: Far col
        :type fare_col: str
        :param age_col: Age col
        :type age_col: str
        :param n_bins: Number of bins, defaults to 4
        :type n_bins: int, optional
        """
        # Bin continuous variables
        self.data['fare_category'] = pd.cut(self.data[fare_col], bins=n_bins, labels=False)
        self.data['age_category'] = pd.cut(self.data[age_col], bins=n_bins, labels=False)

    def _extract_title_from_name(self, name_col: str):
        """Extracts the title from the name column. Also based on:

            - https://www.kaggle.com/imoore/titanic-the-only-notebook-you-need-to-see

        :param name_col: Name col
        :type name_col: str
        """
        # Extract the title from the name column
        self.data['title'] = self.data[name_col].str.extract(' ([A-Za-z]+)\.', expand=False) # (\w+\.) matches the first word ending with a dot character
        self.data['title'] = self.data['title'].str.lower()
                    
        # Handle missing titles
        most_common_title = self.data['title'].mode().iloc[0]
        self.data['title'].fillna(most_common_title)

        # One hot encode the titles
        dummy_titles = pd.get_dummies(self.data['title'], prefix='title')
        self.data = pd.concat([self.data, dummy_titles], axis=1)

    def _extract_ticket_info(self, ticket_col: str):
        """Extracts information from the ticket column. Mostly based on: 
            
            - https://www.kaggle.com/pliptor/titanic-ticket-only-study

        :param ticket_col: Ticket col
        :type ticket_col: str
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
        """Conducts advanced preprocessing and feature engineering

        :param name_col: Name col, defaults to 'Name'
        :type name_col: str, optional
        :param cabin_col: Cabin col, defaults to 'Cabin'
        :type cabin_col: str, optional
        :param sibbling_spouse_col: SibSp col, defaults to 'SibSp'
        :type sibbling_spouse_col: str, optional
        :param parent_child_col: Parch col, defaults to 'Parch'
        :type parent_child_col: str, optional
        :param fare_col: Fare col, defaults to 'Fare'
        :type fare_col: str, optional
        :param age_col: Age col, defaults to 'Age'
        :type age_col: str, optional
        :param ticket_col: Ticket col, defaults to 'Ticket'
        :type ticket_col: str, optional
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