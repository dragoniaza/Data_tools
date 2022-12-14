import uvicorn
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import TraffyFondue_Model
from TraffyFondue_Model import TraffyFondueModel

class Item(BaseModel):
    type: Optional[str] = None
    type_id: Optional[str] = None
    comment: Optional[str] = None
    coords: Optional[list] = None
    district: Optional[str] = None
    subdistrict: Optional[str] = None
    province: Optional[str] = None
    timestamp: Optional[str] = None

app = FastAPI()

@app.post("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(item: Item):
    logistic_model = RandomForestClassifier(random_state = 42)
    traffy = TraffyFondueModel(logistic_model, item)
    preprocess = traffy.import_data()
    trained_model = traffy.train_Model(preprocess)
    response = {"type":trained_model[0]}
    return response

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=5500, reload=True)