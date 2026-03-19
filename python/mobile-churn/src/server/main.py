from fastapi import FastAPI
from pathlib import Path 
from contextlib import asynccontextmanager
PATH_ROOT=Path(__file__).resolve().parent.parent.parent
import sys 
sys.path.append(str(PATH_ROOT))
from src.inference.predictor_factory import PredictorFactory
from src.inference.predictor import TelecomChurnPredictor
from src.server.schemas import ChurnInput


@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.predictor = PredictorFactory.build()
    
    yield

    app.state.predictor = None

app = FastAPI(lifespan=lifespan)

@app.post('/predict')
def inference(input:ChurnInput):
    predictor_instance = app.state.predictor
    response = predictor_instance.predict(input_data=input)
    return response.model_dump(by_alias=True, mode="json")