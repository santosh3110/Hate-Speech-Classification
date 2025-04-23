from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from hate_speech_classifier.pipeline.train_pipeline import run_pipeline
from hate_speech_classifier.pipeline.predict_pipeline import PredictionPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    pipeline = PredictionPipeline()
    prediction = pipeline.run(text)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "input_text": text,
        "prediction": prediction
    })

@app.get("/train")
async def train_model():
    run_pipeline()
    return {"message": "Model training complete!"}
