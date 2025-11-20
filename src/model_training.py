import os
import yaml
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("data.log")
file_handler.setLevel(logging.DEBUG)

consoler = logging.StreamHandler()
consoler.setLevel(logging.DEBUG)

formater = logging.Formatter("%(asctime)s-%(lineno)d-%(levelname)s-%(message)s")

file_handler.setFormatter(formater)
consoler.setFormatter(formater)

logger.addHandler(file_handler)
logger.addHandler(consoler)
logger.debug("model-training module entered")

def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise e
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise e
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise e

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error("train.csv does not exist")
        raise e

def processing(df):
    try:
        x = df.drop(["loan_paid_back"],axis=1).values
        y = df["loan_paid_back"].values
    except Exception as e:
        logger.error("cant convert dataframe to array")
        raise e
    try:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        return x,y
    except Exception as e:
        logger.error("isuue with standard scaler")
        raise e


def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def main():
    path = r"data/clean_data/train.csv"
    df = load_data(path)
    x,y = processing(df)
    logger.debug("preprocessing done")

    params = load_params(params_path='params.yaml')
    randomstate = params['model_training']['random_state']
    nestimators = params['model_training']['n_estimators']
    criterions = params['model_training']['criterion']
    maxdepth = params['model_training']['max_depth']
    min_samples_split = params['model_training']['min_samples_split']
    bootstraps = params['model_training']['bootstrap']
    
    try:
        model = RandomForestClassifier(random_state = randomstate, n_estimators = nestimators, 
                                       criterion = criterions, max_depth = maxdepth,
                                       min_samples_split=min_samples_split, bootstrap = bootstraps)
    except Exception as e:
        logger.error("error in model creation")
        raise e
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 42)

    model.fit(x_train,y_train)
    logger.debug("done model training")

    dir_path = r"models"
    os.makedirs(dir_path, exist_ok = True)
    file_path = os.path.join(dir_path, "model.pkl")
    save_model(model,file_path)

    dir = r"data/evaul_data"
    os.makedirs(dir, exist_ok = True)

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    logger.debug("train testing data extracted")


    file_path = os.path.join(dir, "x_train.csv")
    x_train.to_csv(file_path, index = False)
    file_path = os.path.join(dir, "x_test.csv")
    x_test.to_csv(file_path, index = False)

    file_path = os.path.join(dir, "y_train.csv")
    y_train.to_csv(file_path, index = False)
    file_path = os.path.join(dir, "y_test.csv")
    y_test.to_csv(file_path, index = False)

    logger.debug("training testing data saved")



if __name__ == "__main__":
    main()