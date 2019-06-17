import numpy as np
import pandas as pd
from data.imputers import Imputer, AgeImputer, GoDateImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import KBinsDiscretizer

class Discretizer(TransformerMixin):
    def __init__(self, columns, num_bins=5):
        self.columns = columns
        self.num_bins = num_bins
        self.column_bins = dict().fromkeys(self.columns)
        
    def fit(self, df, y=None):
        for column in self.columns:
            _, bins = pd.cut(df[column], self.num_bins, retbins=True)
            self.column_bins[column] = bins
        return self
    
    def transform(self, df):
        X = df.copy()
        for column in self.columns:
#            print (f'{column}: {self.column_bins[column]}')
            X[column] = pd.cut(X[column], self.column_bins[column], labels=False)
        X.fillna(value=0.0, inplace=True) # assign all out-of-bounds values to bin 0
        return X
    

# missing values imputation pipeline
def impute():
    mean_imp_columns = ['attr', 'sinc', 'fun', 'int', 'amb', 'int_corr',
                        'attr_o', 'sinc_o', 'fun_o', 'int_o', 'amb_o']
#    mode_imp_columns = ['exp_happy', 'exp_happy_o']
    mean_imputer = Imputer(columns=mean_imp_columns, strategy='mean')
#    mode_imputer = Imputer(columns=mode_imp_columns, strategy='most_frequent')
    imputation_pipe = Pipeline([('age_imputer', AgeImputer()),
                                ('go_date_imputer', GoDateImputer()),
                                ('mean_imputer', mean_imputer),
#                                ('mode_imputer', mode_imputer)
                                ])
    return imputation_pipe


def discretize(df):
    numeric_columns = df.select_dtypes(include='float64').columns.tolist()
    discretization_pipe = Pipeline([('discretizer', Discretizer(numeric_columns, num_bins=5))])
    return discretization_pipe

# column encoder pipeline
def encode(df):
    
    # categories in decreasing temporal order (same for all ordinal columns)
    ordinal_categories = [['Almost never',
                           'Several times a year',
                           'Once a month',
                           'Twice a month',
                           'Once a week',
                           'Twice a week',
                           'Several times a week']] * 4

    ordinal_columns = ['go_date', 'go_out', 'go_date_o', 'go_out_o']
#    nominal_columns = ['gender', 'race', 'race_o', 
#                       'goal', 'goal_o', 'field', 'field_o']
    nominal_columns = ['gender', 'race', 'race_o', 'goal', 'goal_o']
                           
    numeric_columns = df.select_dtypes(include='float64').columns.tolist()

    ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)
    nominal_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    
    # delete the first dummy column from each categorical variable
    def handle_dummy_trap(X, df_X=None):
        if df_X is None:
            return X
        indices = [0]
        index = 0
        for i in range(len(nominal_columns) - 1):
            column = nominal_columns[i]
            num_dummies = df_X[column].nunique()
            # print(f'{column} unique values: {num_dummies}')
            index += num_dummies
            indices.append(index)
        X = np.delete(X, indices, axis=1)
        return X

    ordinal_pipe = Pipeline([('encode', ordinal_encoder),
                             ('standardize', RobustScaler())])

    nominal_pipe = Pipeline([('encode', nominal_encoder),
                             ('dummy trap', FunctionTransformer(
                                            handle_dummy_trap,
                                            validate=False,
                                            kw_args={'df_X': df}))
                             ])
                             


    column_transformer = make_column_transformer(
        (nominal_pipe, nominal_columns),
        (ordinal_pipe, ordinal_columns),
        (RobustScaler(), numeric_columns),
#        (KBinsDiscretizer(n_bins=4, encode='ordinal'), numeric_columns),
        remainder='passthrough')

    return column_transformer


def make_pipeline(df):
    imputation_pipe = impute()
    column_transformer = encode(df)
    discretization_pipe = discretize(df)
    transformation_pipe = Pipeline([('impute', imputation_pipe),
#                                    ('discretize', discretization_pipe),
                                   ('encode', column_transformer)
                                   ])
    return transformation_pipe

def make_imputation_pipe():
    imputation_pipe = impute()
    return Pipeline([('impute', imputation_pipe)])

def make_encoding_pipe(df):
    column_transformer = encode(df)
    return Pipeline([('encode', column_transformer)])
