import uvicorn

from libs.data_processing import load_model, data_processing, post_processing
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

@app.post('/predict', tags=["predict"])
def predict(request: RequestJSON):
    df_processed, numeric_transformer, scaled_df = data_processing(request)

    model = load_model();
    prediction = model.predict(df_processed)

    final_result = post_processing(prediction, scaled_df, numeric_transformer)

    return jsonable_encoder({'result': final_result[0][0]})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)