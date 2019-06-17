from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# custom one-hot encoder that takes care of the dummy variable trap
class Dummifier(OneHotEncoder):

    
    def __init__(self, **kwargs):
        super(Dummifier, self).__init__(**kwargs)

    def transform(self, df):
        X = df.copy()
        X = super(Dummifier, self).transform(X)
        return X

