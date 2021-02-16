
from src.training import Model
from src.utils import set_root_logger
from src.preprocessing import Dataset


class MLPipeline:
    def __init__(
        self, 
        df_path_train, 
        df_path_test, 
        id_col, 
        ground_truth, 
        missing_value_config, 
        encoding_config, 
        model_config
    ):
        set_root_logger()
        self.df_path_train = df_path_train
        self.df_path_test = df_path_test
        self.id_col = id_col
        self.ground_truth = ground_truth
        self.col_name_to_fill_method = missing_value_config
        self.col_name_to_encoding = encoding_config
        self.model_name_to_params = model_config

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

    def _run_prep_pipeline(self, advanced_preprocessing=False):
        self.ds.do_basic_preprocessing(self.col_name_to_fill_method, self.col_name_to_encoding)
        cols_to_drop = ['Name', 'Ticket', 'Cabin', 'Embarked']
        self.train_df = self.ds.drop_cols(cols_to_drop, mode='training')
        self.test_df = self.ds.drop_cols(cols_to_drop, mode='testing')

        if advanced_preprocessing: 
            self.ds.do_advanced_preprocessing()
            cols_to_drop += ['title', 'ticket_prefix', 'ticket_group']
            self.train_df = self.ds.drop_cols(cols_to_drop, mode='training')
            self.test_df = self.ds.drop_cols(cols_to_drop, mode='testing')


    def _run_training_pipeline(self, advanced_preprocessing=False):
        for model_name, model_params in self.model_name_to_params.items():
            model = Model(
                model_name=model_name,
                ground_truth='Survived', 
                id_col_name='PassengerId',
                scaling_mode='min_max',
                **model_params
            )
            model.train_and_evaluate(self.train_df)
            if advanced_preprocessing:
                model.gen_submission_file(self.test_df, submission_name='advanced')
            else:
                model.gen_submission_file(self.test_df, submission_name='basic')

    def run(self, advanced_preprocessing):
        self._run_prep_pipeline(advanced_preprocessing)
        self._run_training_pipeline(advanced_preprocessing)