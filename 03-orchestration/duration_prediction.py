#!/usr/bin/env python
# coding: utf-8

import logging
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
import scipy.sparse

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from zenml import step , pipeline

from typing_extensions import Annotated


import mlflow

def save_file(model, path: str):
    """Save the model to a specified path."""
    with open(path, "wb") as f_out:
        pickle.dump(model, f_out)

@step 
def setup_mlflow():
    # Set up MLflow tracking URI and experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Use SQLite for local tracking
    mlflow.set_experiment("nyc-taxi-experiment")

    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)


@step
def read_dataframe(year : int , month:int) -> Annotated[pd.DataFrame, "DataFrame containing taxi trip data"]:
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    

    return df

@step
def create_X(df, dv=None) -> tuple[Annotated[scipy.sparse._csr.csr_matrix, "Feature matrix"], Annotated[DictVectorizer, "Fitted DictVectorizer"]]:
    
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
        save_file(dv, "models/preprocessor.b")
    else:
        X = dv.transform(dicts)

    return X, dv

@step
def extract_target(df: pd.DataFrame, target: str) -> Annotated[pd.Series, "Target column"]:
    return df[target]

@step
def train_model(X_train, y_train, X_val, y_val, dv) -> Annotated[str, "MLflow run ID"]:
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, name="models_mlflow",input_example=X_val[:5].toarray(), registered_model_name="nyc-taxi-duration-prediction")

        return run.info.run_id

@pipeline
def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = extract_target(df_train, "duration")
    y_val = extract_target(df_val, "duration")

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_instance = run(year=args.year, month=args.month)
    #n_id = run_instance()
    #access to output of the pipeline 
    
    """try :
        output = run_instance.outputs
        run_id = output['run_id']
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            

    except Exception as e:
        logging.error(f"error while saving the run_id: {e}")"""