import os
import pickle
from autogluon.tabular import TabularPredictor

def load_model():
    models_path = "ml_models//champion"
    predictor_path = os.path.join(models_path, "predictor.pkl")
    raw_path = os.path.join(models_path, "model.pkl")
    if os.path.isfile(predictor_path):
        champion_predictor = TabularPredictor.load("data/06_models/champion")
        return champion_predictor
    elif os.path.isfile(raw_path):
        with open(raw_path, 'rb') as file:
            champion_model = pickle.load(file)
            return champion_model
    else:
        return ""