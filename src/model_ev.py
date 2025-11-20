import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s-%(lineno)d-%(levelname)s-%(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.debug("entered model-ev stage")

def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test, y_test):
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics, file_path):
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        model_path = r".\models\model.pkl"
        clf = load_model(model_path)
        x_train_path=r".\data\evaul_data\x_train.csv"
        x_test_path=r".\data\evaul_data\x_test.csv"
        y_train_path=r".\data\evaul_data\y_train.csv"
        y_test_path=r".\data\evaul_data\y_test.csv"
        x_train = load_data(x_train_path)
        x_test = load_data(x_test_path)
        y_train = load_data(y_train_path)
        y_test = load_data(y_test_path)

        x_train = x_train.values
        x_test = x_test.values
        y_train = y_train.values
        y_test = y_test.values
        
        mat_train = evaluate_model(clf, x_train, y_train)
        mat_test = evaluate_model(clf, x_test, y_test)

        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy_train', mat_train['accuracy'])
            live.log_metric('precision_train', mat_train['precision'])
            live.log_metric('recall_train', mat_train['recall'])
            live.log_metric('accuracy_test', mat_test['accuracy'])
            live.log_metric('precision_test', mat_test['precision'])
            live.log_metric('recall_test', mat_test['recall'])

            live.log_params(params)
        
        save_metrics(mat_train, 'reports/metrics.json')
        save_metrics(mat_test, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()