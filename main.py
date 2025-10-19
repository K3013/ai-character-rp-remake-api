# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from typing import Optional
from dotenv import load_dotenv

# Gemini client (google genai)
try:
    from google import genai
except Exception as e:
    genai = None

# Diffusers pipeline lazy import
sd_pipeline = None

# Llama local (llama-cpp-python)
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

load_dotenv()

app = FastAPI(title="Hybrid Gemini + Local Models API")

# Config (env)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")         # pour Gemini Developer API
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # alias courant
LOCAL_SD_MODEL = os.environ.get("LOCAL_SD_MODEL", "runwayml/stable-diffusion-v1-5")
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "/models/llama/ggml-model.bin")


class GenRequest(BaseModel):
    mode: str                      # "gemini_text", "gemini_image", "sd_image", "local_qa"
    prompt: str
    # optional params
    guidance: Optional[float] = 7.5
    steps: Optional[int] = 30
    max_tokens: Optional[int] = 512
    user_id: Optional[str] = None

# --- Gemini helpers -------------------------------------------------------
def init_gemini_client():
    global genai
    if genai is None:
        raise RuntimeError("google-genai package not installed")
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
    return client

def gemini_generate_text(prompt: str, max_tokens: int = 512):
    client = init_gemini_client()
    # simple generate_content example (see documentation for richer usage)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    # The genai client returns structured object; use resp.text or resp.output depending on version
    try:
        return resp.text
    except:
        return str(resp)

def gemini_generate_image(prompt: str):
    client = init_gemini_client()
    # Use the image generation APIs (documented at ai.google.dev/gemini-api/docs/image-generation)
    # Example API call shape may vary; using generate_image content pseudo-code:
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, modality="image")
    return resp  # adjust parsing depending on client response

# --- Diffusers helpers ----------------------------------------------------
def init_sd_pipeline():
    global sd_pipeline
    if sd_pipeline is None:
        from diffusers import StableDiffusionPipeline
        import torch
        sd_pipeline = StableDiffusionPipeline.from_pretrained(LOCAL_SD_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            sd_pipeline = sd_pipeline.to("cuda")
    return sd_pipeline

def sd_generate_image(prompt: str, num_inference_steps: int = 30, guidance_scale: float = 7.5, out_path="out.png"):
    pipe = init_sd_pipeline()
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    image.save(out_path)
    return out_path

# --- Local LLM (llama.cpp) helpers ---------------------------------------
_llama_instance = None
def init_llama():
    global _llama_instance
    if Llama is None:
        raise RuntimeError("llama-cpp-python not installed")
    if _llama_instance is None:
        _llama_instance = Llama(model_path=LLAMA_MODEL_PATH)
    return _llama_instance

def local_qa(prompt: str):
    llm = init_llama()
    resp = llm(prompt, max_tokens=256)
    return resp.get("choices", [{"text": ""}])[0]["text"]

# --- API endpoints -------------------------------------------------------
@app.post("/generate")
async def generate(req: GenRequest):
    if req.mode == "gemini_text":
        try:
            text = gemini_generate_text(req.prompt, max_tokens=req.max_tokens)
            return {"mode": req.mode, "output": text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if req.mode == "gemini_image":
        try:
            resp = gemini_generate_image(req.prompt)
            # NOTE: depends on gemini client: could return base64 or url. Return raw object for now.
            return {"mode": req.mode, "output": str(resp)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if req.mode == "sd_image":
        try:
            out = sd_generate_image(req.prompt, num_inference_steps=req.steps or 30, guidance_scale=req.guidance or 7.5)
            return {"mode": req.mode, "output_file": out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if req.mode == "local_qa":
        try:
            out = local_qa(req.prompt)
            return {"mode": req.mode, "output": out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=400, detail="Unknown mode")

@app.get("/health")
async def health():
    return {"status":"ok"}
