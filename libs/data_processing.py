import os
import pickle
import pandas as pd

from autogluon.tabular import TabularPredictor

def load_model():
    models_path = "ml_models//champion"
    predictor_path = os.path.join(models_path, "predictor.pkl")
    raw_path = os.path.join(models_path, "model.pkl")
    if os.path.isfile(predictor_path):
        champion_predictor = TabularPredictor.load("ml_models/champion")
        return champion_predictor
    elif os.path.isfile(raw_path):
        with open(raw_path, 'rb') as file:
            champion_model = pickle.load(file)
            return champion_model
    else:
        return ""

def load_label_encoders():
    with open("ml_models/champion/label_encoders.pkl", "rb") as le_file:
        return pickle.load(le_file)

def load_numeric_transformer():
    with open("ml_models/champion/numeric_transformer.pkl", "rb") as nt_file:
        return pickle.load(nt_file)

def data_processing(request):
    label_encoders = load_label_encoders()
    numeric_transformer = load_numeric_transformer()

    df = pd.DataFrame(request.__dict__, index=[0])
    df['math_score'] = 0
    columns_order = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course','math_score', 'reading_score', 'writing_score']
    df = df[columns_order]

    num_features = df.select_dtypes(exclude="object").columns
    cat_features = df.select_dtypes(include="object").columns

    scaled_df = numeric_transformer.transform(df[num_features])
    scaled_df = pd.DataFrame(scaled_df, columns=num_features)

    encoded_df = []
    for col in cat_features:
        encoded_col = label_encoders[col].transform(df[col])
        encoded_df.append(pd.DataFrame(encoded_col, columns=[col]))

    encoded_df = pd.concat(encoded_df, axis=1)
    df_processed = pd.concat([scaled_df, encoded_df], axis=1)
    df_processed = df_processed.drop(columns=['math_score'])

    return df_processed, numeric_transformer, scaled_df

def post_processing(prediction, scaled_df, numeric_transformer):
    df_post_processing = pd.DataFrame(prediction, columns=["math_score"])
    df_post_processing = pd.concat([df_post_processing, scaled_df.drop(columns=['math_score'])], axis=1) 
    final_df = numeric_transformer.inverse_transform(df_post_processing)

    return final_df