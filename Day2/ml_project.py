import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder=None
    print("waring: category_encoder not installed ")
    
def main():
    print("Loading Dataset..")
    file_path='train.csv'

    if not os.path.exists(file_path):
        return
    else:
        df=pd.read_csv(file_path)
        print(f"file found and loaded!\n Rows:{df.shape[0]},Features:{df.shape[1]}")

    df.loc[0:25,'H']=np.nan
    imputer=SimpleImputer(strategy='median')

    df['H'] = imputer.fit_transform(df[['H']])
    print(df['H'].isnull())


if __name__=="__main__":
    main()