
import json
import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


import config_manager
config_manager.init_config()


import request_manager

import adapter_ollama
import adapter_groq
import adapter_openai
import adapter_mistral
import adapter_anthropic
import adapter_together
import adapter_lmstudio

ROUTE_DB = {
    "OLLAMA": adapter_ollama.process_request,
    "GROQ": adapter_groq.process_request,
    "OPENAI": adapter_openai.process_request,
    "MISTRAL": adapter_mistral.process_request,
    "ANTHROPIC":adapter_anthropic.process_request,
    "TOGETHER": adapter_together.process_request,
    "LMSTUDIO": adapter_lmstudio.process_request  
}

def get_adapter_route(provider):
    return ROUTE_DB.get(provider, None)


app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config_manager.APP_CONFIG["allowed_origins"],
    allow_credentials=True,
    allow_methods=["*"],  # Or specify methods e.g., ["GET", "POST"]
    allow_headers=["*"],  # Or specify headers
)


async def stream_response_data(response_data):
    # Convert the entire response object to a JSON string.
    json_str = json.dumps(response_data)
    
    # Define the size of each chunk you want to stream. This is arbitrary and can be adjusted.
    chunk_size = 1024  # Adjust based on your needs or testing.

    # Stream the JSON string in chunks.
    for i in range(0, len(json_str), chunk_size):
        yield "data: " + json_str[i:i+chunk_size]
        await asyncio.sleep(0.1)  # Simulate delay for streaming; adjust as necessary.

    # Optionally, signal the end of the stream if needed.
    yield ""  # Signal the end of the stream, if your client-side logic requires it.


# Dependency for API key authorization
async def verify_api_key(request: Request):
    global ERROR_AUTH_RESPONSE
    global API_KEYS
    if config_manager.APP_CONFIG["auth_enforcement_enabled"]:
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail=request_manager.ERROR_AUTH_RESPONSE)
        token_type, _, api_key = authorization.partition(' ')
        if token_type.lower() != "bearer" or api_key not in API_KEYS:
            raise HTTPException(status_code=401, detail=request_manager.ERROR_AUTH_RESPONSE)

async def get_header_info(request_headers):
    header_info = {
        "llm_provider": request_headers.get("LLM_PROVIDER", config_manager.APP_CONFIG["default_provider"]).upper(),
    }    
    provider_auth = request_headers.get("Authorization", None)
    if provider_auth is not None:
        provider_auth = provider_auth.replace("Bearer ", "")

    if request_headers.get("PROVIDER_AUTH") is not None:
        provider_auth = request_headers.get("PROVIDER_AUTH")

    header_info["provider_auth"] = provider_auth
    if "MAX_CONTEXT" in request_headers:
        header_info["max_context"] = request_headers["MAX_CONTEXT"]
    return header_info


@app.post("/v1/chat/completions")
async def handle_completions(request: Request,_=Depends(verify_api_key)):
    header_info = await get_header_info(request.headers)
    process_request = get_adapter_route(header_info['llm_provider'])
    if process_request is None:
        raise HTTPException(status_code=400, detail=request_manager.ERROR_PROVIDER_RESPONSE)

    try:
        request_body = await request.json()
    except:
        raise HTTPException(status_code=400, detail=request_manager.ERROR_BAD_REQUEST)
    
    stream_response = request_body.get("stream", False)
    
    # Assuming non-streaming fetch from the provider          
    response = await process_request(request.url.path, header_info, request_body)
    if response.success is False:
        raise HTTPException(status_code=response.status_code, detail=response.body)

    if stream_response:
        # Create a StreamingResponse from an async generator
        return StreamingResponse(stream_response_data(response.body))
    else:
        # If not streaming, return the response normally
        return response.body


# -- EMBEDDINGS ROUTING ---
@app.post("/v1/embeddings")
async def get_embeddings(request: Request,_=Depends(verify_api_key)):
    header_info = await get_header_info(request.headers)
    process_request = get_adapter_route(header_info['llm_provider'])
    if process_request is None:
        raise HTTPException(status_code=400, detail=request_manager.ERROR_PROVIDER_RESPONSE)

    try:
        request_body = await request.json()
    except:
        raise HTTPException(status_code=400, detail=request_manager.ERROR_BAD_REQUEST)

    response = await process_request(request.url.path, header_info, request_body)
    if response.success is False:
        raise HTTPException(status_code=response.status_code, detail=response.body)
    
    return response.body

# --- MODELS ROUTING ---

# List all available models.
@app.get("/v1/models")
async def get_models(request: Request, _=Depends(verify_api_key)):
    header_info = await get_header_info(request.headers)
    process_request = get_adapter_route(header_info['llm_provider'])
    if process_request is None:
        raise HTTPException(status_code=400, detail=request_manager.ERROR_PROVIDER_RESPONSE)

    response = await process_request(request.url.path, header_info, None) 
    if response.success is False:
        raise HTTPException(status_code=response.status_code, detail=response.body)
    return response.body

# Get information about a specific model.
@app.get("/v1/models/{model_id}")
async def get_model(request: Request, model_id: str, _=Depends(verify_api_key)):
    header_info = await get_header_info(request.headers)
    process_request = get_adapter_route(header_info['llm_provider'])
    if process_request is None:
        raise HTTPException(status_code=400, detail=request_manager.ERROR_PROVIDER_RESPONSE)

    response = await process_request(request.url.path, header_info, None)
    if response.success is False:
        raise HTTPException(status_code=response.status_code, detail=response.body)
    
    return response.body


if __name__ == "__main__":

    uvicorn.run(app, host=config_manager.APP_CONFIG['host'], port=config_manager.APP_CONFIG['port'])