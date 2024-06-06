import uvicorn
import pickle
import pandas as pd

from libs.load_model import load_model
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

class RequestJSON(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

app = FastAPI()

# @app.get("/", tags=["intro"])
# def index():
#     return {"message": "Linear Regression ML"}

@app.post('/predict', tags=["predict"])
def predict(request: RequestJSON):
    model = load_model();

    with open("ml_models/champion/label_encoders.pkl", "rb") as le_file:
        label_encoders = pickle.load(le_file)
    with open("ml_models/champion/numeric_transformer.pkl", "rb") as nt_file:
        numeric_transformer = pickle.load(nt_file)

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

    prediction = model.predict(df_processed)

    print(prediction)

    result = pd.DataFrame(prediction, columns=["math_score"])

    print(result)

    result = pd.concat([result, scaled_df.drop(columns=['math_score'])], axis=1) 

    print(result)

    final_result = numeric_transformer.inverse_transform(result)

    return jsonable_encoder({'result': final_result[0][0]})



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)