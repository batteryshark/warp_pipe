import json
import base64
import struct
import time
import httpx
import dirtyjson

import config_manager
import request_manager
import oai_tools


# Pull the provider specific options or set defaults if they don't exist already.
ADAPTER_CONFIG = config_manager.get_provider_options("LMSTUDIO", {"base_url": "http://localhost:1234", "api_key":""})
    
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

async def process_function_calling(request_headers, request_body):

    is_streaming_response = request_body.get("stream", False)

    """
    Sends a simulated function calling request to an LLM via the /api/chat endpoint,
    including a system prompt that details available functions.
    """
    # Validate presence of tools in the request
    if "tools" not in request_body:
        response = request_manager.ResponseStatus(400, {"error": "No tools for function calling specified in the request."})
        return response

    # Construct the system prompt explaining the available functions and their parameters
    system_prompt = "You have the following functions available to you:\n"
    for tool in request_body.get("tools", []):
        function_info = tool.get("function", {})
        function_name = function_info.get("name")
        parameters = function_info.get("parameters", {})

        system_prompt += f"- Function Name: {function_name}, Parameters: {json.dumps(parameters)}\n"
    
    system_prompt += "Please execute any function you deem appropriate based on the context provided.\n"
    system_prompt += """ Respond only in a valid JSON block containing the following keys: \n
    "name": "function_name", \n
    "arguments": { "parameter1": "value1", "parameter2": "value2" } \n
    """

    # Add this system prompt to the message history for the /api/chat request
    if "messages" not in request_body:
        request_body["messages"] = []

    if len(request_body["messages"]) == 0:
        response = request_manager.ResponseStatus(400, {"error": "No user messages found in the request."})
        return response        

    request_body["messages"].append({
        "role": "system",
        "content": system_prompt
    })

    # Convert OpenAI request format to OLLAMA request format for /api/chat endpoint
    request_body = {
        "model":request_body['model'],
        "response_format": { "type": "json_object" },
        "stream": False,
        "messages": request_body.get("messages", []),
        "temperature": 0
        # Assuming your existing conversion logic is applied here
    }

    

    # Send the request to the LLM
    print("WARN: Sending Tool Request - This is SUPER Experimental!")
    url, headers = await construct_request(request_headers, "/v1/chat/completions")
    mistral_response = await request_manager.send_request("POST", url, headers=headers, body=request_body)
   

    # Validate the LLM's response
    if mistral_response.status_code != 200:
        # Handle error scenarios appropriately
        openai_response = request_manager.ResponseStatus(mistral_response.status_code, mistral_response.body)


    prompt_tokens = mistral_response.body.get("prompt_eval_count",0)
    completion_tokens = mistral_response.body.get("eval_count",0)

    total_tokens = prompt_tokens + completion_tokens
 
    tool_calls = []
    message_content = mistral_response.body["choices"][0]["message"]['content']
    if "```" in message_content:
        message_content = message_content.replace("```json","").replace("```","")
    try:
        response_content = dirtyjson.loads(message_content)
    except:
        response_content = {}

    if "name" in response_content:
        if "arguments" in response_content:
            tool_calls.append({
                "index":0,
                "id": f"call_{int(time.time())}",
                "type":"function",
                "function":{
                "name": response_content["name"],
                "arguments": json.dumps(response_content["arguments"])
                }
            })
    # Assume the LLM understood and "executed" the function by including its output in the response
    # Format the response in OpenAI's function calling format
    message_key = "message"
    object_type = "chat.completion"
    if is_streaming_response:
        message_key = "delta"
        object_type = "chat.completion.chunk"
    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": object_type,
        "created": int(time.time()),
        "model": mistral_response.body.get("model",request_body['model']),
        "system_fingerprint": "fp_1234567890",  # This can be a hash of the response content
        "choices": [
            {
                "index": 0,
                message_key: {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                },
                "logprobs": None,
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,  # Update based on actual usage
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }

    openai_response = request_manager.ResponseStatus(200, response)
    openai_response.success = True
    return openai_response

async def chat_completions(request_headers, request_body):

    is_streaming_response = request_body.get("stream", False)

    # We will need this later.
    number_of_completions = request_body.get("n", 1)
    openai_response = request_manager.ResponseStatus(0, None)

    provider_request = {
        'model': request_body['model'],
        'messages': request_body['messages'],
    }
    if "response_format" in request_body:
        provider_request["response_format"] = request_body["response_format"]
    if "temperature" in request_body:
        provider_request["temperature"] = request_body["temperature"]
    if "top_p" in request_body:
        provider_request["top_p"] = request_body["top_p"]
    if "max_tokens" in request_body:
        provider_request["max_tokens"] = request_body["max_tokens"]
    
    if "seed" in request_body:
        for i in range(0,len(provider_request["messages"])):
            provider_request["messages"][i]["seed"] == request_body['seed']

    if "stop" in request_body:
        print("WARNING: Only using the first stop paramter")
        provider_request["stop"] = request_body["stop"][0]

    if "tools" in request_body:
        provider_request["tools"] = request_body["tools"]
    
    if "tool_choice" in request_body:
        provider_request["tool_choice"] = request_body["tool_choice"]

    if "tools" in request_body:
        if not request_body['model'].startswith("mistral-large"):
            return await process_function_calling(request_headers, request_body)

    response_messages = []
    prompt_tokens = 0
    completion_tokens = 0
    response_content = {}
    for i in range(0,number_of_completions):
        url, headers = await construct_request(request_headers, "/v1/chat/completions")   
        response = await request_manager.send_request("POST", url, headers, provider_request)
        
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
        elif not "choices" in response.body:
            print("Messages not found in response")
            openai_response.status_code = 500
            openai_response.body = request_manager.ERROR_UNKNOWN_ERROR
            return openai_response
        response_messages.append(response.body["choices"][0])
        prompt_tokens += response.body['usage'].get("prompt_tokens",0)
        completion_tokens += response.body['usage'].get("completion_tokens",0)
        response_content = response.body

    total_tokens = prompt_tokens + completion_tokens

    for i in range(0,len(response_messages)):
        response_messages[i]['index'] = i
    response_content["choices"] = response_messages
    response_content["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

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
    
async def get_embeddings(request_headers, request_body):
    if "dimensions" in request_body:
        print("WARNING: Dimensions parameter is not supported. Ignoring.")
    
    input_list = request_body["input"]
    encoding_format = request_body.get("encoding_format", "float")
    if not isinstance(input_list, list):
        input_list = [input_list]

    mistral_body = {
        'model': request_body['model'],
        'input': input_list,
        'encoding_format':'float'
    }

    url, headers = await construct_request(request_headers, "/v1/embeddings")
    response = await request_manager.send_request("POST",url, headers=headers, body=mistral_body)
    if response.status_code != 200:
        return response
    
    response_content = response.body
    converted_data = []
    if encoding_format == "base64":
        for data in response_content["data"]:
            if data['object'] != "embedding":
                converted_data.append(data)
                continue
            data['embedding'] = oai_tools.encode_embeddings_to_base64(data['embedding'])
            converted_data.append(data)

    openai_response = request_manager.ResponseStatus(response.status_code, response_content)
    openai_response.success = True
    return openai_response

async def list_models(request_headers, request_body):
    url, headers = await construct_request(request_headers, "/v1/models")

    provider_response = await request_manager.send_request('GET',url,headers)
    if provider_response.status_code != 200:
        if provider_response.status_code == 404:
            provider_response.response.body = request_manager.ERROR_INVALID_REQUEST
        elif provider_response.status_code == 401:
            provider_response.body = request_manager.ERROR_AUTH_RESPONSE
        elif provider_response.status_code == 400:
            provider_response.body = request_manager.ERROR_BAD_REQUEST
        elif provider_response.status_code == 500:
            provider_response.body = request_manager.ERROR_INTERNAL_SERVER_ERROR
        else:
            provider_response.body = request_manager.ERROR_UNKNOWN_ERROR
        return provider_response

    # convert_datetime_to_epoch
    openai_response_body = {
        "object": "list",
        "data": []
    }

    for model in provider_response.body['data']:
        model['id'] = model['id'].split("/")[-1]
        openai_response_body["data"].append({
            "id": model["id"],
            "object": "model",
            "created": 0,
            "owned_by": model['owned_by']
        })

    openai_response = request_manager.ResponseStatus(200, openai_response_body)
    openai_response.success = True
    return openai_response   
    
async def get_model(request_headers,request_body):
    # This is a little gross because we have to list all models to get any details.
    created_time = 0
    owner = "organization-owner"
    list_response = await list_models(request_headers,request_body)
    openai_response = request_manager.ResponseStatus(0, None)

    if list_response.success is False:
        openai_response.body = request_manager.ERROR_INTERNAL_SERVER_ERROR
        openai_response.status_code = 500
        return openai_response
    
    provider_models = list_response.body
    model_exists = False
    for model in provider_models['data']:
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