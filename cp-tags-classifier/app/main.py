from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from tree_sitter import Language, Parser
from astminer import extract_tokens, extract_ast_path2, extract_ast_path
import tensorflow as tf
from attention import Attention
from custom import CustomCrossEntropy
from pydantic import BaseModel
import uvicorn
import pickle
import os

app = FastAPI()
templates = Jinja2Templates(directory=".")


@app.get("/demo", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})


from_disk = pickle.load(open("./model/vectorize_token.pkl", "rb"))
encoder = tf.keras.layers.TextVectorization.from_config(from_disk["config"])
encoder.set_weights(from_disk["weights"])
model = tf.keras.models.load_model(
    "./model/tok.h5", custom_objects={"CustomCrossEntropy": CustomCrossEntropy}
)
inputs = tf.keras.Input(shape=(1,), dtype="string")
indices = encoder(inputs)
outputs = model(indices)
end_to_end_model = tf.keras.Model(inputs, outputs)


class SourceCodeIn(BaseModel):
    text: str


class TokenOut(BaseModel):
    tokens: list


class TagOut(BaseModel):
    data_structure: float
    graph: float
    math: float
    string: float
    geometry: float


CPP_LANGUAGE = Language("./model/tree-sitter-cpp-language.so", "cpp")
PARSER = Parser()
PARSER.set_language(CPP_LANGUAGE)


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/code-token", response_model=TokenOut)
def tokenize(payload: SourceCodeIn):
    tokens = extract_tokens(PARSER, payload.text)
    return {"tokens": tokens}


@app.post("/ast-path", response_model=TokenOut)
def tokenize(payload: SourceCodeIn):
    tokens = extract_ast_path2(PARSER, payload.text)
    return {"tokens": tokens}


@app.post("/predict", response_model=TagOut)
def tokenize(payload: SourceCodeIn):
    src = payload.text
    res = list(end_to_end_model.predict([" ".join(extract_tokens(PARSER, src))]))[0]
    res = list(res)

    return {
        "data_structure": res[0],
        "graph": res[1],
        "math": res[2],
        "string": res[3],
        "geometry": res[4],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('HTTP_PORT') or 8001))