import os
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, callbacks

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

def build_model(input_dim):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  
    return model

def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def main():
    path=r"data/clean_data/train.csv"
    df=load_data(path)
    x,y=processing(df)
    
    try:
        model=build_model(x.shape[1])
    except Exception as e:
        logger.error("error in model creation")
        raise e
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    es = callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=4, mode='max', verbose=1)
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test,y_test),
        epochs=100,
        batch_size=2048,
        callbacks=[es, reduce_lr],
        class_weight=class_weights,
        verbose=0
    )
    dir_path = r"models"
    os.makedirs(dir_path, exist_ok = True)
    file_path = os.path.join(dir_path, "model.pkl")
    save_model(model,file_path)


if __name__ == "__main__":
    main()