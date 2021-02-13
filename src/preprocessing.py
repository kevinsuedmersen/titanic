import pandas as pd

class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        print(F'Read dataframe from {filepath} into memory')
