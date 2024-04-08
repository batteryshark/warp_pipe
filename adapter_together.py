import json
import base64
import struct
import time
import httpx

import config_manager
import request_manager
import oai_tools


# Pull the provider specific options or set defaults if they don't exist already.
ADAPTER_CONFIG = config_manager.get_provider_options("TOGETHER", {"base_url": "https://api.together.xyz", "api_key":""})
    
async def construct_request(request_headers, endpoint):
    api_key = ADAPTER_CONFIG["api_key"]
    if request_headers != None and request_headers.get("provider_auth"):
        api_key = request_headers.get("provider_auth")    
    headers = {
        "Authorization": "Bearer " + api_key,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }    
    url = f"{ADAPTER_CONFIG['base_url']}{endpoint}"
    return url, headers


async def chat_completions(request_headers,request_body):
   
    is_streaming_response = request_body.get("stream", False)
    request_body['stream'] = False

    url,headers= await construct_request(request_headers, "/v1/chat/completions")
    response = await request_manager.send_request("POST", url, headers,request_body)
    if response.status_code != 200:
        return response
    
    response_content = response.body      
    if is_streaming_response:
        response_content["object"] = "chat.completion.chunk"
        stream_choices = []
        
        for choice in response_content["choices"]:
            tool_index = 0
            for i in range(0,len(choice['message']["tool_calls"])):
                choice['message']['tool_calls'][i]['index'] = tool_index
                tool_index += 1
            choice['delta'] = choice['message']
            del choice['message']
            stream_choices.append(choice)
        response_content["choices"] = stream_choices
        print(response_content)
    
    openai_response = request_manager.ResponseStatus(response.status_code, response_content)
    openai_response.success = True
    return openai_response
    
async def get_embeddings(request_headers,request_body):
    url,headers= await construct_request(request_headers, "/v1/embeddings")
    openai_response = await request_manager.send_request("POST", url, headers,request_body)
    if openai_response.status_code == 200:
        openai_response.success = True
    return openai_response    

async def list_models(request_headers, request_body):    
    url,headers= await construct_request(request_headers, "/v1/models")
    openai_response = await request_manager.send_request("GET", url, headers)
    if openai_response.status_code == 200:
        for i in range(0,len(openai_response.body)):
            openai_response.body[i]["id"] = openai_response.body[i]["id"].split("/")[-1]
        openai_response.success = True
    return openai_response    
    
async def get_model(request_headers, request_body):
    # This is a little gross because we have to list all models to get any details.
    created_time = 0
    owner = "organization-owner"
    list_response = await list_models(request_headers, request_body)
    openai_response = request_manager.ResponseStatus(0, None)

    if list_response.success is False:
        openai_response.body = request_manager.ERROR_INTERNAL_SERVER_ERROR
        openai_response.status_code = 500
        return openai_response

    list_of_models = list_response.body
    model_exists = False
    for model in list_of_models:
        if request_body['model_id'] == model["id"]:
            created_time = model["created"]
            owner = model["organization"]
            model_exists = True
            break

    # We'll handle the error message in the main code.
    if not model_exists:
        openai_response.body = request_manager.ERROR_MODEL_NOT_FOUND
        openai_response.status_code = 404
        return openai_response
    
    openai_response_body = {
        'id': request_body['model_id'],
        'object': 'model',
        'created': created_time,
        'owned_by':  owner
    }
    openai_response.body = openai_response_body
    openai_response.success = True
    openai_response.status_code = 200
    return openai_response
        
# -- ROUTING --

async def process_request(request_type, request_headers, request_body):    
    # Completions API Handling
    if request_type == "/v1/chat/completions":
        return await chat_completions(request_headers, request_body)
    # Embeddings API Handling
    elif request_type == "/v1/embeddings":
        return await get_embeddings(request_headers, request_body)
    # Model API Handling
    elif request_type == "/v1/models":
        return await list_models(request_headers, request_body)
    elif request_type.startswith("/v1/models/"):
        model_id = request_type.split("/")[-1]
        if request_body == None:
            request_body = {}
        request_body["model_id"] = model_id
        return await get_model(request_headers,request_body)    
    else:
        return request_manager.ResponseStatus(400, request_manager.ERROR_NOT_IMPLEMENTED)
    
