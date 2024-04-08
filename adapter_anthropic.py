import json
import base64
import struct
import time
import httpx

import config_manager
import request_manager
import oai_tools


# Pull the provider specific options or set defaults if they don't exist already.
ADAPTER_CONFIG = config_manager.get_provider_options("ANTHROPIC", {"base_url": "https://api.anthropic.com", "api_key":""})
    
async def construct_request(request_headers, endpoint):
    api_key = ADAPTER_CONFIG["api_key"]
    if request_headers != None and request_headers.get("provider_auth"):
        api_key = request_headers.get("provider_auth")    
    headers = {
        "x-api-key": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "anthropic-version":"2023-06-01",
        "anthropic-beta": "tools-2024-04-04"
    }    
    url = f"{ADAPTER_CONFIG['base_url']}{endpoint}"
    return url, headers

def convert_openai_request_to_anthropic(openai_request):
    # Initialize the base structure of the Anthropic request
    anthropic_request = {
        "model": openai_request.get("model", "").replace("gpt-3.5-turbo", "claude-3-opus-20240229"),
        "max_tokens": 1024,  # Assuming a default; adjust as necessary
        "tools": [],
        "messages": openai_request.get("messages", [])
    }
    
    # Convert tools from OpenAI to Anthropic format
    for tool in openai_request.get("tools", []):
        if tool.get("type") == "function":
            function = tool.get("function", {})
            anthropic_tool = {
                "name": function.get("name"),
                "description": function.get("description"),
                "input_schema": {
                    "type": "object",
                    "properties": function.get("parameters", {}).get("properties", {}),
                    "required": function.get("parameters", {}).get("required", [])
                }
            }
            # In Anthropic API, tools don't directly support a "unit" parameter as OpenAI might, so we'll omit it
            anthropic_request["tools"].append(anthropic_tool)
    
    # Assuming "tool_choice" does not have a direct equivalent in Anthropic, it will be ignored.
    
    return anthropic_request

def convert_anthropic_response_to_openai(anthropic_response):
    # Initialize the base structure for the OpenAI response
    openai_response = {
        "id": anthropic_response.get("id", "").replace("msg_", "chatcmpl-"),  # Example transformation
        "object": "chat.completion",
        "created": 1699896916,  # Placeholder, real timestamp generation would be required
        "model": anthropic_response.get("model", "claude-3-haiku"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,  # Will fill this later if there's text content
                    "tool_calls": []
                },
                "logprobs": None,
                "finish_reason": "tool_calls"  # Assuming tool use; adjust as necessary
            }
        ],
        "usage": {
            "prompt_tokens": anthropic_response.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": anthropic_response.get("usage", {}).get("output_tokens", 0),
            "total_tokens": 0  # Calculate later
        }
    }

    # Process each item in the Anthropic response content
    for item in anthropic_response.get("content", []):
        if item.get("type") == "text":
            # Assuming only one text content for simplicity; append or adjust as needed
            openai_response["choices"][0]["message"]["content"] = item.get("text")
        elif item.get("type") == "tool_use":
            tool_call = {
                "id": item.get("id").replace("toolu_", "call_"),  # Example ID transformation
                "type": "function",
                "function": {
                    "name": item.get("name"),
                    "arguments": str(item.get("input")).replace("'", "\"")  # Convert dict to JSON-like string
                }
            }
            openai_response["choices"][0]["message"]["tool_calls"].append(tool_call)
    
    # Calculate total tokens
    openai_response["usage"]["total_tokens"] = openai_response["usage"]["prompt_tokens"] + openai_response["usage"]["completion_tokens"]
    
    return openai_response

async def chat_completions(request_headers,request_body):
   
    is_streaming_response = request_body.get("stream", False)
    request_body['stream'] = False
    if 'tools' in request_body:
        request_body = convert_openai_request_to_anthropic(request_body)
        url, headers = await construct_request(request_headers, "/v1/messages")   
        response = await request_manager.send_request("POST", url, headers, request_body)
        if response.status_code == 200:
            response.success = True
            response.body = convert_anthropic_response_to_openai(response.body)
        return response
            
    # Separate System Message from Messages
    anthropic_messages = []
    system_message = None
    json_mode = False
    if "response_format" in request_body:
        json_mode = True
        del request_body["response_format"]
    
    for message in request_body['messages']:
        if message['role'] == "system":
            system_message = message['content']
        else:
            anthropic_messages.append(message)
    request_body['messages'] = anthropic_messages
    if not 'max_tokens' in request_body:
        request_body['max_tokens'] = 4096
    
    if system_message is not None:
        request_body['system'] = system_message
    if json_mode:
        if not 'system' in request_body:
            request_body['system'] = ''
        request_body['system'] += "\n Output Format: Strictly in JSON."
    
    response_messages = []
    prompt_tokens = 0
    completion_tokens = 0

    # We will need this later.
    number_of_completions = request_body.get("n", 1)
    if 'n'  in request_body:
        del request_body['n']    
    openai_response = request_manager.ResponseStatus(0, None)

    for i in range(0,number_of_completions):
        url, headers = await construct_request(request_headers, "/v1/messages")   
        response = await request_manager.send_request("POST", url, headers, request_body)

        openai_response.status_code = response.status_code
        if response.status_code == 400:
            if "model is required" in str(response.body):
                openai_response.body = request_manager.ERROR_MODEL_NOT_FOUND
            else:
                openai_response.body = request_manager.ERROR_BAD_REQUEST
            return openai_response
        elif response.status_code == 500:
            openai_response.body = request_manager.ERROR_INTERNAL_SERVER_ERROR
            return openai_response
        elif response.status_code != 200:
            openai_response.status_code = 500
            openai_response.body = request_manager.ERROR_UNKNOWN_ERROR
            return openai_response
        elif not "content" in response.body:
            print("Messages not found in response")
            openai_response.status_code = 500
            openai_response.body = request_manager.ERROR_UNKNOWN_ERROR
            return openai_response
        response_message = {
            "id": response.body["id"],
            "logprobs": None,
            "role": response.body["role"],
            "finish_reason":"stop"
        }
        if response.body['content'][0]['type'] == "text":
            response_message['content'] = response.body['content'][0]['text']
        else:
            response_message['content'] = response.body['content'][0]

        response_messages.append(response_message)
        prompt_tokens += response.body['usage'].get("input_tokens",0)
        completion_tokens += response.body['usage'].get("output_tokens",0)

    total_tokens = prompt_tokens + completion_tokens
    
    # Converting the Messages Back to OpenAI Format
    openai_response_messages = []

    for i in range(0,len(response_messages)):
        message = response_messages[i]
        openai_message = {
            "index": i,
            "logprobs": None,
            "finish_reason": "stop"
        }
        message_key = "message"
        response_object = "chat.completion"
        if is_streaming_response:
            response_object = "chat.completion.chunk"
            message_key = "delta"
        
        openai_message[message_key] = {
            "role": message["role"],
            "content": message["content"]
        }

        openai_response_messages.append(openai_message)

    openai_response = {
        "id": "chatcmpl-123",
        "object": response_object,
        "created": int(time.time()),
        "model": request_body["model"],
        "system_fingerprint": "fp_44709d6fcb",
        "choices": openai_response_messages,
        "usage":{
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }
    response.body = openai_response
    response.success = True
    response.status_code = 200
    return response
    
async def list_models(request_headers, request_body):    
    ### TODO: Implement actual API Polling - They Hardcode it into their SDK so I don't feel that bad about this
    response = {
        "object": "list",
        "data":[    
            {
                "id": "claude-3-opus-20240229",
                "object": "model",
                "created": 0,
                "owned_by": "Anthropic"
            },            
            {
                "id": "claude-3-sonnet-20240229",
                "object": "model",
                "created": 0,
                "owned_by": "Anthropic"
            },            
            {
                "id": "claude-3-haiku-20240307",
                "object": "model",
                "created": 0,
                "owned_by": "Anthropic"
            },    
            {
                "id": "claude-2.1",
                "object": "model",
                "created": 0,
                "owned_by": "Anthropic"
            },    
            {
                "id": "claude-2.0",
                "object": "model",
                "created": 0,
                "owned_by": "Anthropic"
            },    
            {
                "id": "claude-instant-1.2",
                "object": "model",
                "created": 0,
                "owned_by": "Anthropic"
            }                                                    
        ]
    }
    openai_response = request_manager.ResponseStatus(200, response)
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
    
    list_of_models = list_response.body['data']
    model_exists = False
    for model in list_of_models:
        if model["id"] == request_body['model_id']:
            created_time = model["created"]
            owner = model["owned_by"]
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
        return request_manager.ResponseStatus(400, request_manager.ERROR_NOT_IMPLEMENTED)
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