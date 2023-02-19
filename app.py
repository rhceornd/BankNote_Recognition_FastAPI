import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

pickle_in = open("../bankNote/classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return { "Message" : "ML banknote recognition API"}


@app.post("/predict")
def predict_banknote(data : BankNote): # BankNote에서 정의한 데이터 모양 읽음
    data = data.dict()
    variance = data["variance"]
    skewness = data['skewness']
    curtosis = data["curtosis"]
    entropy = data["entropy"]

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if (prediction[0] > 0.5):
        result = "flase paper"
    else:
        result = "true paper"

    return {"prediction" : result}



if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)


# uvicorn app : app --reload