
import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import os 

#Catefory_encoders is needed for Target Encoding text 

try:
    from category_encoders import TargetEncoder
except ImportError:
    print("Warning: categoty_encoders not installed. Target Encoding will not be used")

def main():
    print("Loading Datasets...")
    file_path = os.path.join(os.path.dirname(__file__), "train.csv")

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return

    df=pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully.\nRows:{df.shape[0]}\nFeatures:{df.shape[1]}")
    print(df.head())
    print(df.columns)

    #Handling missing values 
    print("\n--- Handling Missing Values ---")
    print("Artificially deleting some'Hits'(H) data demonstrate.")

    df.loc[0:25,'H']=np.nan

    imputer=SimpleImputer(strategy='median')

    df['H']=imputer.fit_transform(df[['H']])
    print(f"Imputation complete. 'Hits' (H) now has {df['H'].isnull().sum() } null values")

    #SKEWED Distribution
    print("Evaluating the skewness of the Runs(R) distribution...")

    #Apply np.logic (logarithmic + 1) to compress the distribution of numbers 

    df['LogRuns']=np.log1p(df['R'])
    print(f"Log Transformation applied. New skewness:{df['LogRuns'].skew():.2f}(closer to 0 is perfectly balanced).\n")
    print(df.head())
    df['Team_ID'] = ['Team_' + str(np.random.randint(1,150)) for _ in range(len(df))]
                
    if TargetEncoder is not None:
        print("Applying Target Encoder")
                    
        encoder = TargetEncoder()
                    
        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'],df['W'])
                
    else:
        print("category encoder not installed")

    features_to_test =['R', 'HR','SO','SB']

    x_features=df[features_to_test].fillna(0)
    y_target=df['W']

    selector=SelectKBest(score_func=mutual_info_regression,k=2)
    selector.fit_transform(x_features,y_target)
    winning_features=selector.get_support()
    best_features=x_features.columns[winning_features].tolist()

    print(best_features)

    x=df[best_features]
    y=df['W']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    print(f"training data size{x_train.shape}\ntesting data size:{x_test.shape}\n")

    #training the model
    model=LinearRegression()
    model.fit(x_train,y_train)

    prediction=model.predict(x_test)
    print(prediction)

    actual_wins=y_test.head(3).values
    predicted_wins=prediction[:3]

    for i in range(3):
        predicted=round(predicted_wins[i])
        actual=actual_wins[i]
        difference=abs(actual-predicted)
        
        print(f"Model Gussed:{predicted}")
        print(f"real answer:{actual}")
        print(f"differences:{difference}")

if __name__=='__main__':
    main()
        
            
        