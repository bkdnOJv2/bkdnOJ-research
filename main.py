from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory=".")


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

class PredictData(BaseModel):
    code: str

@app.post("/api/v0/predict")
async def predict(request: Request, data: PredictData):
    code = data.code
    return {
        'code': 200,
        'message': "Ok",
        'data': {
            'graph': 0.1235981,
            'string': 0.4392456,
            'math': 0.0001351,
            'data_structure': 0.012389,
            'geometry': 0.00581928,
        }
    }
