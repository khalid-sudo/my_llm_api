from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.llm.model import generate_response
import torch

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 512

class GenerationResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerationResponse)
async def generate(prompt_request: PromptRequest):
    try:
        response = generate_response(prompt_request.prompt)
        return GenerationResponse(response=response)
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=503,
            detail="Not enough memory to process the request. Try reducing the prompt length."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )