import os
import shutil
import logging
from IPython.display import display
from src.training import Model
from src.utils import set_root_logger
from src.preprocessing import Dataset

logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(
        self, 
        df_path_train: str, 
        df_path_test: str, 
        id_col: str, 
        ground_truth: str,
        results_dir: str='results'
    ):
        """
        :param df_path_train: Path to training dataset
        :param df_path_test: Path to test dataset
        :param id_col: Name of the id column
        :param results_dir: Dir where results are stored
        :param clear_results: Whether or not to delete results from prior runs
        :return: self
        """
        set_root_logger()
        self.df_path_train = df_path_train
        self.df_path_test = df_path_test
        self.id_col = id_col
        self.ground_truth = ground_truth
        self.results_dir = results_dir
        self._prepare_workspace()
        self.ds = Dataset(
            df_path_train=df_path_train,
            df_path_test=df_path_test,
            id_col=id_col,
            ground_truth=ground_truth
        )
        self.train_df = None
        self.test_df = None

    def _prepare_workspace(self):
        """Prepares the workspace, i.e. removes/creates folders on disk
        """
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
            logger.info('Cleared results dir')
        os.makedirs(self.results_dir)
        logger.info('Created results dir')

    def run_eda(self, title: str='ds_profile_report', html_path:str='results/ds_profile_report.html'):
        """Runs exploratory data analysis
        :param title: Title of pandas profiling report
        :html_path: Path to pandas profiling report
        :return: None
        """
        self.ds.profile(title=title, html_path=html_path)

    def _run_prep_pipeline(self, missing_value_config: dict, encoding_config: dict, advanced_preprocessing: bool=False):
        """Always conducts basic preprocessing and optionally also conducts advanced preprocessing
        :param missing_value_config: Dictionary containing information how to deal with missing values
        :param encoding_config: Dictionary containing information how to deal with categorical variables
        :return: None
        """
        self.ds.do_basic_preprocessing(missing_value_config, encoding_config)
        cols_to_drop = ['Name', 'Ticket', 'Cabin', 'Embarked']
        self.train_df = self.ds.drop_cols(cols_to_drop, mode='training')
        self.test_df = self.ds.drop_cols(cols_to_drop, mode='testing')

        if advanced_preprocessing: 
            self.ds.do_advanced_preprocessing()
            cols_to_drop += ['title', 'ticket_prefix', 'ticket_group']
            self.train_df = self.ds.drop_cols(cols_to_drop, mode='training')
            self.test_df = self.ds.drop_cols(cols_to_drop, mode='testing')

        print('train dataframe:')
        display(self.train_df)
        print('\ntest dataframe:')
        display(self.test_df)

    def _run_training_pipeline(self, advanced_preprocessing: bool, model_config: dict, hp_tuning: bool):
        """Trains and evaluates all models specified in model_config. Scaling input features is also
        optionally conducted
        :param advanced_preprocessing: Whether or not to conduct advanced preprocessing
        :param model_config: Model configuration
        :param hp_tuning: Whether or not to perform hyper parameter tuning
        :return: None
        """
        for model_name, model_params in model_config.items():

            # Temporarily remove the scaling mode from the model config
            scaling_mode = model_params.pop('scaling_mode')
            
            # Conduct a single training run or hp tuning
            if not hp_tuning:
                model = Model(
                    model_name=model_name,
                    ground_truth='Survived', 
                    id_col_name='PassengerId',
                    scaling_mode=scaling_mode,
                    **model_params
                )    
                model.train_and_evaluate(self.train_df)
                submission_filepath = f'{model_name}_single_training'
            else:
                model = Model(
                    model_name=model_name,
                    ground_truth='Survived', 
                    id_col_name='PassengerId',
                    scaling_mode=scaling_mode
                )
                model.hparam_tuning(self.train_df, model_params)
                submission_filepath = f'{model_name}_hp_tuning'
            
            # Attach the sacling mode again
            model_params['scaling_mode'] = scaling_mode

            # Save submission file
            if advanced_preprocessing:
                submission_filepath += '_advanced_prep.csv'
                model.gen_submission_file(self.test_df, submission_filepath=submission_filepath)
            else:
                submission_filepath += '_basic_prep.csv'
                model.gen_submission_file(self.test_df, submission_filepath=submission_filepath)

    def run(
        self, 
        missing_value_config: dict, 
        encoding_config: dict, 
        advanced_preprocessing: bool, 
        model_config: dict=None, 
        hp_tuning: bool=False
    ):
        """Runs the preprocessing and training sub-pipelines
        :param missing_value_config: Configuration how to deal with missing values
        :param encoding_config: Configuration how to encode categorical variables
        :param advanced_preprocessing: Whether or not to conduct advanced preprocessing
        :param model_config: Model configuration
        :param hp_tuning: Whether or not to perform hyper parameter tuning
        """
        self._run_prep_pipeline(missing_value_config, encoding_config, advanced_preprocessing)
        self._run_training_pipeline(advanced_preprocessing, model_config, hp_tuning)