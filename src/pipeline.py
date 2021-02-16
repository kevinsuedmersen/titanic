from IPython.display import display
from src.training import Model
from src.utils import set_root_logger
from src.preprocessing import Dataset


class MLPipeline:
    def __init__(
        self, 
        df_path_train, 
        df_path_test, 
        id_col, 
        ground_truth
    ):
        set_root_logger()
        self.df_path_train = df_path_train
        self.df_path_test = df_path_test
        self.id_col = id_col
        self.ground_truth = ground_truth

        self.ds = Dataset(
            df_path_train=df_path_train,
            df_path_test=df_path_test,
            id_col=id_col,
            ground_truth=ground_truth
        )
        self.train_df = None
        self.test_df = None

    def run_eda(self, title='ds_profile_report', html_path='results/ds_profile_report.html'):
        self.ds.profile(title=title, html_path=html_path)

    def _run_prep_pipeline(self, missing_value_config, encoding_config, advanced_preprocessing=False):
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

    def _run_training_pipeline(self, advanced_preprocessing, model_config):
        for model_name, model_params in model_config.items():
            scaling_mode = model_params.pop('scaling_mode')
            model = Model(
                model_name=model_name,
                ground_truth='Survived', 
                id_col_name='PassengerId',
                scaling_mode=scaling_mode,
                **model_params
            )
            model_params['scaling_mode'] = scaling_mode
            model.train_and_evaluate(self.train_df)
            
            if advanced_preprocessing:
                model.gen_submission_file(self.test_df, submission_name='advanced')
            else:
                model.gen_submission_file(self.test_df, submission_name='basic')

    def run(self, missing_value_config, encoding_config, advanced_preprocessing, model_config):
        self._run_prep_pipeline(missing_value_config, encoding_config, advanced_preprocessing)
        self._run_training_pipeline(advanced_preprocessing, model_config)