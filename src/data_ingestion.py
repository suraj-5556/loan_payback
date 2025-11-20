import mysql.connector
import pandas as pd
import logging
from dotenv import load_dotenv
import os

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
logger.debug("data-ingestion module entered")



def fetch_cred ():
    try:
        load_dotenv() 
    except Exception as e:
        logger.error("env file not exist")
        raise e
    try:
        host = os.getenv("DB_HOST")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        database = os.getenv("DB_NAME")
        table = os.getenv("DB_TABLE")
        return host,user,password,database,table
    except Exception as e:
        logger.error("credentials not found in .env file")
        raise e
    


def fetch_data(host,user,password,database,table):
    try:
        conn = mysql.connector.connect(
        host = host,
        user = user,
        password = password,
        database = database)
    except Exception as e:
        logger.error("failed to bild connection between sql")
        raise e
    try:
        query = f"SELECT * FROM {table};"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error("failed to load data from sql")
        raise e
    

def main():
    host,user,password,database,table=fetch_cred()
    df = fetch_data(host,user,password,database,table)
    logger.debug("data retrival from sql")
    dir_path = "data/raw"
    os.makedirs(dir_path, exist_ok = True)
    file_path = os.path.join(dir_path, "train.csv")
    df.to_csv(file_path, index = False)


if __name__ == "__main__":
    main()