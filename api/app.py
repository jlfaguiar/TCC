from fastapi import FastAPI
from pydantic import BaseModel, Field
from pipeline_functions import classificar_sentimento
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Permite todas as origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Você pode restringir isso depois
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class EntradaTexto(BaseModel):
    texto: str = Field(..., example="Esse jogo é uma porcaria")
    temperatura: float = Field(0.3, ge=0.0, le=2, example=0.5)  # limite até 2 por precaução

@app.post("/classificar/")
def classificar(dado: EntradaTexto):
    sentimento = classificar_sentimento(dado.texto, temperatura=dado.temperatura)
    return {"classe": sentimento}

