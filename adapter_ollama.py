
import json
import time
import dirtyjson

import config_manager
import request_manager
import oai_tools

# Pull the provider specific options or set defaults if they don't exist already.
ADAPTER_CONFIG = config_manager.get_provider_options("OLLAMA", {"base_url": "http://localhost:11434", "model_settings":{}})
    
async def construct_request(request_headers, endpoint):
    api_key = ADAPTER_CONFIG.get("api_key",None)
    if request_headers != None and request_headers.get("provider_auth"):
        api_key = request_headers.get("provider_auth")    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }    
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{ADAPTER_CONFIG['base_url']}{endpoint}"
    return url, headers

async def process_function_calling(selected_model, is_streaming_response, request_headers, openai_request_body):
    """
    Sends a simulated function calling request to an LLM via the /api/chat endpoint,
    including a system prompt that details available functions.
    """
    # Validate presence of tools in the request
    if "tools" not in openai_request_body:
        response = request_manager.ResponseStatus(400, {"error": "No tools for function calling specified in the request."})
        return response

    # Construct the system prompt explaining the available functions and their parameters
    system_prompt = "You have the following functions available to you:\n"
    for tool in openai_request_body.get("tools", []):
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
    if "messages" not in openai_request_body:
        openai_request_body["messages"] = []

    if len(openai_request_body["messages"]) == 0:
        response = request_manager.ResponseStatus(400, {"error": "No user messages found in the request."})
        return response        

    openai_request_body["messages"].append({
        "role": "system",
        "content": system_prompt
    })

    # Convert OpenAI request format to OLLAMA request format for /api/chat endpoint
    ollama_request_body = {
        "model":selected_model,
        "format": "json",
        "stream": False,
        "messages": openai_request_body.get("messages", []),
        "options":{
            "temperature": 0
        }
        # Assuming your existing conversion logic is applied here
    }

    

    # Send the request to the LLM
    print("WARN: Sending Tool Request to OLLAMA - This is SUPER Experimental!")
    url,headers = await construct_request(request_headers, "/api/chat")
    ollama_response = await request_manager.send_request("POST", url,headers, body=ollama_request_body)
   
    

    # Validate the LLM's response
    if ollama_response.status_code != 200:
        # Handle error scenarios appropriately
        openai_response = request_manager.ResponseStatus(ollama_response.status_code, ollama_response.body)


    prompt_tokens = ollama_response.body.get("prompt_eval_count",0)
    completion_tokens = ollama_response.body.get("eval_count",0)

    total_tokens = prompt_tokens + completion_tokens

    tool_calls = []
    try:
        response_content = dirtyjson.loads(ollama_response.body["message"]['content'])
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
        "model": openai_request_body.get("model", "unknown-model"),
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
    if not "model" in request_body:
        return request_manager.ResponseStatus(400, request_manager.ERROR_BAD_REQUEST)
    is_streaming_response = request_body.get("stream", False)

    num_ctx = 0
    num_gpu = None
    model_settings = ADAPTER_CONFIG["model_settings"].get(request_body["model"], {})
    model_name = request_body["model"]
    if model_settings != {}:
        model_name = model_settings["model"]
        if 'num_ctx' in model_settings:
            num_ctx = model_settings["num_ctx"]
        if 'num_gpu' in model_settings:
            num_gpu = model_settings["num_gpu"]

    # Override if we got a custom value from the header.
    if "max_context" in request_body:
        num_ctx = request_body["max_context"]

    ollama_request_body = {
        'model': model_name,
        # In the interest of having ONE stream only, we're going to always disable streaming for this.
        'stream': False,
        'messages': []
    }
    if "response_format" in request_body:
        if request_body["response_format"]["type"] == "json_object":
            ollama_request_body["format"] = "json"

    if "tools" in request_body or "tool_choice" in request_body:
        return await process_function_calling(model_name, is_streaming_response,request_headers, request_body)
    
    ollama_options = {}

    if num_ctx > 0:
        ollama_options["num_ctx"] = num_ctx

    if num_gpu is not None:
        ollama_options["num_gpu"] = num_gpu

    if "frequency_penalty" in request_body:
        ollama_options["repeat_penalty"] = request_body["frequency_penalty"]

    if "logit_bias" in request_body:
        print("WARNING: Logit bias is not supported by OLLAMA. Ignoring.")
    
    if "logprobs" in request_body:
        print("WARNING: Logprobs is not supported by OLLAMA. Ignoring.")

    if "top_logprobs" in request_body:
        print("WARNING: Top logprobs is not supported by OLLAMA. Ignoring.")

    if "seed" in request_body:
        ollama_options["seed"] = request_body["seed"]

    if "top_p" in request_body:
        ollama_options["top_p"] = request_body["top_p"]

    if "max_tokens" in request_body:
        ollama_options["num_predict"] = request_body["max_tokens"]

    if "temperature" in request_body:
        ollama_options["temperature"] = request_body["temperature"]

    if "stop" in request_body:
        if(len(request_body["stop"]) > 1):
            print("WARNING: Multiple stop tokens are not supported by OLLAMA. Using the first one.")
        ollama_options["stop"] = request_body["stop"][0]

    # We will need this later.
    number_of_completions = request_body.get("n", 1)

    # Time to convert the messages.
    ollama_messages = []
    for message in request_body.get("messages", []):
        ollama_message = {
            "role": message["role"],            
        }
        if isinstance(message["content"], str):
            ollama_message["content"] = message["content"]
        elif isinstance(message["content"], list):
            for content in message["content"]:
                if content['type'] == "text":
                    ollama_message['content'] = content['text']
                    break
            
            for content in message['content']:
                if content['type'] == "image_url":
                    url = content["image_url"]['url']
                    # Download and base64 image from url
                    image_b64 = await oai_tools.download_image_from_url_and_encode_b64(url)
                    if not 'images' in ollama_message:
                        ollama_message['images'] = []
                    ollama_message['images'].append(image_b64)
           
        ollama_messages.append(ollama_message)
    
    ollama_request_body["messages"] = ollama_messages


    response = request_manager.ResponseStatus(0, None)
    ollama_response_messages = []
    prompt_tokens = 0
    completion_tokens = 0

    for i in range(0,number_of_completions):
        url, headers = await construct_request(request_headers, "/api/chat")
        ollama_response = await request_manager.send_request("POST",url,headers, body=ollama_request_body)
        response.status_code = ollama_response.status_code
        if ollama_response.status_code == 400:
            if "model is required" in str(ollama_response):
                response.body = request_manager.ERROR_MODEL_NOT_FOUND
            else:
                response.body = request_manager.ERROR_BAD_REQUEST
            return response
        elif ollama_response.status_code == 500:
            response.body = request_manager.ERROR_INTERNAL_SERVER_ERROR
            return response
        elif ollama_response.status_code != 200:
            response.status_code = 500
            response.body = request_manager.ERROR_UNKNOWN_ERROR
            return response
        elif not "message" in ollama_response.body:
            print("Messages not found in response")
            response.status_code = 500
            response.body = request_manager.ERROR_UNKNOWN_ERROR
            return response
        ollama_response_messages.append(ollama_response.body["message"])
        # Ollama's server handles caching of the prompt so if you've already asked this prompt it will not send the eval count because it's not evaluating the prompt again.
        prompt_tokens += ollama_response.body.get("prompt_eval_count",0)
        completion_tokens += ollama_response.body["eval_count"]

    total_tokens = prompt_tokens + completion_tokens

    # Converting the Messages Back to OpenAI Format
    openai_response_messages = []

    for i in range(0,len(ollama_response_messages)):
        message = ollama_response_messages[i]
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
        "model": model_name,
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


async def get_embeddings(request_headers, request_body):
    if "dimensions" in request_body:
        print("WARNING: Dimensions parameter is not supported by OLLAMA. Ignoring.")
    
    input_list = request_body["input"]
    encoding_format = request_body.get("encoding_format", "float")
    if not isinstance(input_list, list):
        input_list = [input_list]
    openai_response = {
        "object": "list",
        "data": [],
        "model": request_body["model"],
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }

    for input_text in input_list:
        ollama_request = {
            "model": request_body["model"],
            "prompt": input_text            
        }
        url,headers = await construct_request(request_headers, "/api/embeddings")
        response = await request_manager.send_request("POST",url,headers,ollama_request)
        if response.status_code == 400:        
            if "model is required" in str(response.body):
                response.body = request_manager.ERROR_MODEL_NOT_FOUND
            else:
                response.body = request_manager.ERROR_BAD_REQUEST
            return response
        elif response.status_code == 500:
            response.body = request_manager.ERROR_INTERNAL_SERVER_ERROR
            return response
        elif response.status_code != 200:
            response.status_code = 500
            response.body = request_manager.ERROR_UNKNOWN_ERROR        
            return response
        elif not "embedding" in response.body:
            print("Embedding not found in response")
            response.status_code = 500
            response.body = request_manager.ERROR_UNKNOWN_ERROR
            return response
        
        embedding_data = response.body["embedding"]
        if encoding_format == "base64":
            embedding_data = oai_tools.encode_embeddings_to_base64(embedding_data)
            
        openai_response["data"].append({
            "object": "embedding",
            "embedding": embedding_data,
            "index": len(openai_response["data"])
        })
    
    response.body = openai_response
    response.success = True
    response.status_code = 200
    return response

async def list_models(request_headers, request_body):
    url, headers = await construct_request(request_headers, "/api/tags")

    ollama_response = await request_manager.send_request('GET',url,headers)
    if ollama_response.status_code != 200:
        if ollama_response.status_code == 404:
            ollama_response.response.body = request_manager.ERROR_INVALID_REQUEST
        elif ollama_response.status_code == 401:
            ollama_response.body = request_manager.ERROR_AUTH_RESPONSE
        elif ollama_response.status_code == 400:
            ollama_response.body = request_manager.ERROR_BAD_REQUEST
        elif ollama_response.status_code == 500:
            ollama_response.body = request_manager.ERROR_INTERNAL_SERVER_ERROR
        else:
            ollama_response.body = request_manager.ERROR_UNKNOWN_ERROR
        return ollama_response

    # convert_datetime_to_epoch
    openai_response_body = {
        "object": "list",
        "data": []
    }

    for model in ollama_response.body['models']:
        openai_response_body["data"].append({
            "id": model["name"],
            "object": "model",
            "created": oai_tools.convert_datetime_to_epoch(model["modified_at"]),
            "owned_by": "organization-owner"
        })

    openai_response = request_manager.ResponseStatus(200, openai_response_body)
    openai_response.success = True
    return openai_response    

# -- MODEL HANDLERS --
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
    
    ollama_models = list_response.body
    model_exists = False
    for model in ollama_models["data"]:
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