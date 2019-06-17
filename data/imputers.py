from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np


# generic custom imputer
class Imputer(TransformerMixin):

    def __init__(self, columns, strategy='mean'):
        self.columns = columns
        self.strategy = strategy
        self.column_imputers = dict()

    def fit(self, df, y=None):
        for column in self.columns:
            imputer = SimpleImputer(strategy=self.strategy)
            imputer.fit(df[[column]])
            self.column_imputers[column] = imputer
        return self

    def transform(self, df):
        X = df.copy()
        for column in self.columns:
            X[column] = self.column_imputers[column].transform(df[[column]])
        return X


# age, age_o: impute based on the mode of age difference
class AgeImputer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        df_m = df[df['gender'] == 'Male']
        self.mode_diff_m = (df_m['age'] - df_m['age_o']).mode()[0].astype(int)
        self.mode_diff_f = -self.mode_diff_m
        return self

    def transform(self, df):
        X = df.copy()
        X['age'] = np.where((X['gender'] == 'Male') & (X['age'].isnull()),
                             X['age_o'] + self.mode_diff_m, df['age'])
        X['age'] = np.where((X.gender == 'Female') & (X['age'].isnull()),
                             X['age_o'] + self.mode_diff_f, X['age'])
        X['age_o'] = np.where((X['gender'] == 'Male') & (X['age_o'].isnull()),
                               X['age'] + self.mode_diff_f, X['age_o'])
        X['age_o'] = np.where((X['gender'] == 'Female') & (X['age_o'].isnull()),
                               X['age'] + self.mode_diff_m, X['age_o'])
        return X


# go_date imputer based on go_out field
class GoDateImputer(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y=None):
        # all missing go_date entries have go_out='Twice a week'
        # these two fields are expected to be correlated
        mask = df['go_out'] == 'Twice a week'
        self.mode = df[mask].go_date.mode()[0]
        return self

    def transform(self, df):
        X = df.copy()
        X.go_date.where(X.go_date.notnull(), other=self.mode, inplace=True)
        X.go_date_o.where(X.go_date_o.notnull(), other=self.mode, inplace=True)
        return X
    