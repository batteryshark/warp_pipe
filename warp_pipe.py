
import json
import asyncio
import time

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

def split_string_by_length(text, end):
    return [text[i:i+end] for i in range(0,len(text),end)]

# OpenAI has a very specific chunk setup it needs and various apis evaluate it differently
# so it has to match EXACTLY... let's do that globally.
def generate_response_chunks(response_data):
    response_chunks = []
    chat_id = response_data['id']
    created_time = int(time.time())
    selected_model = response_data['model']
    system_fingerprint = "warp-pipe-001"
    # TODO - Actually chunk this out so it streams all pretty.
    response_message = response_data['choices'][0]['message']
    response_content = response_message['content']

    # First chunk has no content
    first_response_message = response_message
    first_response_message['content'] = ""

    i_chunk = {
        'id':chat_id,
        'object':'chat.completion.chunk',
        'created':created_time,
        'model':selected_model,
        'system_fingerprint':system_fingerprint,
        'choices':[{
            "index":0,
            "delta":first_response_message,
            "logprobs":None,
            "finish_reason":None
    }]}

    response_chunks.append(json.dumps(i_chunk))

    if response_content and len(response_content) > 1:
        c_content = split_string_by_length(response_content,4096)
        for cc in c_content:
            c_chunk = {
                'id':chat_id,
                'object':'chat.completion.chunk',
                'created':created_time,
                'model':selected_model,
                'system_fingerprint':system_fingerprint,
                'choices':[{
                    "index":0,
                    "delta":{"content":cc},
                    "logprobs":None,
                    "finish_reason":None
                    }
                ]
            }
            response_chunks.append(json.dumps(c_chunk))

    # Yup - it does this.
    final_chunk = {"id":chat_id,"object":"chat.completion.chunk","created":created_time,"model":selected_model,"system_fingerprint":system_fingerprint,"choices":[{"index":0,"delta":{},"logprobs":None,"finish_reason":"stop"}]}
    response_chunks.append(json.dumps(final_chunk))

    # It also does this.
    response_chunks.append("[DONE]")
    return response_chunks

async def stream_response_data(response_data):
    response_chunks = generate_response_chunks(response_data)
    print("Response Chunks")
    for rc in response_chunks:
        print(f"data: {rc}\n\n")

    for chunk in response_chunks:
        yield f"data: {chunk}\n\n"
        #await asyncio.sleep(0.1)


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
    # Provider Auth Passthrough 
    provider_auth = request_headers.get("PROVIDER_AUTH","")
    if provider_auth == "":
        provider_auth = request_headers.get("Authorization", None)
        auth_skip_list = ["Bearer", "Bearer ", "Bearer sk-xxx"]
        if provider_auth in auth_skip_list:
            provider_auth = ""
        else:
            provider_auth = provider_auth.replace("Bearer ","")
    
    if provider_auth != "":
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
        return StreamingResponse(stream_response_data(response.body),media_type='text/event-stream')
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
