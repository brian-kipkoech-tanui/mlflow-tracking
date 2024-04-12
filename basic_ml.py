import os
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
import uuid
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore")


time_day = datetime.utcnow() + timedelta(hours=3)
formatted_time_day  = time_day.strftime('%Y-%m-%d-%H-%M')
unique_id = str(uuid.uuid4().hex)[:6]
unique_format = formatted_time_day + '-' + unique_id
run_name_model_version = "GBR_Regression-V" + unique_format
print("Run name: ", run_name_model_version)


def load_data():
    try:
       train_data = pd.read_csv("https://github.com/brian-kipkoech-tanui/galaxies-wellbeing/blob/master/Data/Train_data.csv")
       test_data = pd.read_csv("https://github.com/brian-kipkoech-tanui/galaxies-wellbeing/blob/master/Data/Validation.csv")
       return train_data, test_data
    except Exception as e:
       raise e
   
def preprocess_data():
    train_data, testing_data = load_data()
    def missing_percentage(df):
        # Count the number of missing values in each column
        missing_values = df.isnull().sum()

        # Total number of rows in the dataframe
        total_rows = len(df)

        # Calculate the percentage of missing values in each column
        percentage_missing_values = (missing_values / total_rows) * 100
        return  percentage_missing_values
    
    train_missing = missing_percentage(train_data)
    test_missing = missing_percentage(testing_data)
    
    train_missing = train_missing.reset_index().rename(columns={'index': 'column', 0: 'missing%'})
    
    cols = train_missing[train_missing['missing%']>70]['column'].to_list()
    train_data.drop(cols, axis=1, inplace=True)
    testing_data.drop(cols, axis=1, inplace=True)
    
    # Use code 0 to mean unknown 
    # For Population using at least basic drinking-water services (%)
    # And Population using at least basic sanitation services (%)
    train_data['Population using at least basic drinking-water services (%)'].fillna(0, inplace=True)
    train_data['Population using at least basic sanitation services (%)'].fillna(0, inplace=True)

    # Test Data
    testing_data['Population using at least basic drinking-water services (%)'].fillna(0, inplace=True)
    testing_data['Population using at least basic sanitation services (%)'].fillna(0, inplace=True)
    
    # Replace missing values with median for all columns
    train_data.fillna(train_data.median(), inplace=True)

    testing_data.fillna(testing_data.median(), inplace=True)
    
    testing_data.drop(['ID', 'Predicted Well-Being Index'], axis=1, inplace=True)

    testing_data = testing_data.copy()
    
    return train_data, testing_data

  
def eval(actual, pred) :
    # rmse = mean_squared_error(actual, pred, squared=False)
    rmse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 =  r2_score(actual, pred)
      
    return rmse, mae, r2
   


def main():
    train_data, testing_data = preprocess_data()
    
    int_cols = train_data.iloc[:,1:].select_dtypes(include=['float', 'int']).columns.to_list()
    # Create the features and label
    X_train = train_data.iloc[:,:-1]
    y_train = train_data.iloc[:,-1].values
    y_train = y_train.reshape(len(y_train), 1)
    X_train.drop('ID', axis=1, inplace=True)
    
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    # Standardise the numerical columns
    X_train[int_cols[:-1]] = sc.fit_transform(X_train[int_cols[:-1]])
    
    test_int_cols = testing_data.drop('galaxy', axis=1).columns.to_list()
    testing_data[test_int_cols] = sc.transform(testing_data[test_int_cols])
    
    # Encoding Categorical Variables
    X_train = pd.get_dummies(X_train, drop_first=True)

    testing_data = pd.get_dummies(testing_data, drop_first=True)
    
    # Split the data into training and testing datasets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    # Sklearn Pipeline to Preprocess and fit the model
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Assuming X_train is a sparse matrix
    std_scaler = StandardScaler(with_mean=False).fit(X_train)

    def preprocessor(X):
        D = np.copy(X)
        D = std_scaler.transform(D)
        return D
    
    from sklearn.preprocessing import FunctionTransformer

    preprocessor_transformer = FunctionTransformer(preprocessor)
    
    from sklearn.pipeline import Pipeline
    # from sklearn.linear_model import LinearRegression
    # from sklearn.neighbors import KNeighborsRegressor as KNR
    # from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor

    p1 = Pipeline([('scaler', preprocessor_transformer),
                ('GBR Regression', GradientBoostingRegressor())])
    
    from sklearn.metrics import mean_absolute_error, r2_score

    def fit_and_print(p, X_train=X_train, X_test=X_val, y_train=y_train, y_test=y_val):
        # Fit the transformer
        p.fit(X_train, y_train)
        # Predict the train and test outputs
        training_prediction = p.predict(X_train)
        test_prediction = p.predict(X_test)
        y_pred = test_prediction
        
        # Calculate Mean Absolute Error
        train_mae = mean_absolute_error(training_prediction, y_train)
        test_mae = mean_absolute_error(test_prediction, y_test)
        
        # Calculate R-squared
        train_r2 = r2_score(y_train, training_prediction)
        test_r2 = r2_score(y_test, test_prediction)
        
        # Plot and save the graph
        plt.scatter(y_test, test_prediction)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.savefig("actual_vs_predicted.png")  # Save the plot
        plt.close()  # Close the plot to avoid displaying in the notebook
        
        # Print the errors
        print("Training Mean Absolute Error: {:.2f}".format(train_mae))
        print("Training R-squared: {:.2f}%".format(train_r2 * 100))
        print("Test Mean Absolute Error: {:.2f}".format(test_mae))
        print("Test R-squared: {:.2f}%".format(test_r2 * 100))
        return p, y_pred, train_mae
    
    # Fit and print a the model
    p, y_pred, train_mae = fit_and_print(p1)
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Galaxy-Well-Being")
    with mlflow.start_run(run_name = run_name_model_version):
        mlflow.log_param("Train-mae", train_mae)
        # Log the graph as an artifact
        mlflow.log_artifact("actual_vs_predicted.png")
        
        rmse, mae, r2= eval(y_val, y_pred)
        
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(p, "trained_model") # model, foldername
        # os.makedirs("dummy", exist_ok=True)
        # with open("dummy/example.txt", "wt") as f:
        #     f.write(f"Artifact created at {time.asctime()}")
        # mlflow.log_artifacts("dummy")
        mlflow.sklearn.save_model(p,
                                  os.path.join("Model" ,
                                               "Artifacts",
                                               unique_format)# Path to save the model Artifacts
                                  )
        
        
        
        
if __name__ == "__main__":
    # args = argparse.ArgumentParser()
    # args.add_argument("--alpha", "-a", type=float, default=0.2)
    # args.add_argument("--l1-ratio", "-l1", type=float, default=0.3)
    # parser_args = args.parse_args()
    
    main()