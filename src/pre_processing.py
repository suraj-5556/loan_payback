import pandas as pd
import os
import logging

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
logger.debug("pre-processing module entered")


def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error("train.csv does not exist")
        raise e


def feature_manager (df):
    try:
        df = pd.get_dummies(df,columns = ['gender','marital_status','education_level','employment_status','loan_purpose'],drop_first=False)

    except Exception as e:
        logger.error("column missmatch")
        raise e
    try:
        dfv= df.drop(["id","gender_Other","marital_status_Widowed","education_level_Other","employment_status_Unemployed","loan_purpose_Other"],axis=1)

    except Exception as e:
        logger.error("cant drop columns")
        raise e
    bool_cols = df.select_dtypes(include = 'bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def process_grade (df):
    try:
        ordered_grade = sorted(df["grade_subgrade"].unique())
        reverse_grade = ordered_grade[::-1]
    except Exception as e:
        logger.error("grade column missmatch")
        raise e
    label_map = {grade: idx for idx, grade in enumerate(reverse_grade)}
    df["grade_subgrade"] = df["grade_subgrade"].map(label_map)
    return df


def main():
    path = r"data\train.csv"
    df = load_data(path)
    df = feature_manager(df)
    df = process_grade(df)
    dir_path = r"data\clean_data"
    os.makedirs(dir_path, exist_ok = True)
    file_path = os.path.join(dir_path, "train.csv")
    df.to_csv(file_path, index = False)


if __name__ == "__main__":
    main()