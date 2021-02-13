import pandas as pd
from pandas_profiling import ProfileReport


class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        print(F'Read dataframe from {filepath} into memory')

    def profile(self, title='Dataset profile report'):
        profile = ProfileReport(self.df, title=title)
        profile.to_notebook_iframe()
