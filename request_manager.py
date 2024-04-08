import httpx
import json

ERROR_AUTH_RESPONSE = {
    "error": {
        "message": "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY), or as the password field (with blank username) if you're accessing the API from your browser and are prompted for a username and password.",
        "type": "invalid_request_error",
        "param": None,
        "code": None
    }
}

ERROR_PROVIDER_RESPONSE = {
    "error": {
        "message": "Invalid LLM Provider specified.",
        "type": "invalid_llm_provider",
        "param": None,
        "code": None
    }
}
ERROR_UNKNOWN_ERROR = {
    "error": {
        "message": "Unknown Error.",
        "type": "unknown_error",
        "param": None,
        "code": None
    }
}

ERROR_INTERNAL_SERVER_ERROR = {
    "error": {
        "message": "Internal Server Error.",
        "type": "internal_server_error",
        "param": None,
        "code": None
    }
}

ERROR_NOT_IMPLEMENTED = {
    "error": {
        "message": "Not Implemented.",
        "type": "not_implemented",
        "param": None,
        "code": None
    }
}

ERROR_BAD_REQUEST = {
    "error": {
        "message": "Invalid Request Body.",
        "type": "invalid_request_body",
        "param": None,
        "code": None
    }
}

ERROR_INVALID_REQUEST = {
  "error": {
    "type": "invalid_request_error",
    "code": "unknown_url",
    "message": "Unknown request URL. Please check the URL for typos.",
    "param": None
  }
}

ERROR_MODEL_NOT_FOUND = {
    "error": {
        "message": "The model does not exist or you do not have access to it.",
        "type": "invalid_request_error",
        "param": None,
        "code": "model_not_found"
    }
}

class ResponseStatus:
    def __init__(self, status_code=500, body=None):
        self.status_code = status_code
        self.success = False    
        self.body = body


async def send_request(method, url, headers={}, body={},cert=None):    
    print(f"Sending Request to: {url}")    
    if not "Content-Type" in headers:
        headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=None,verify=cert) as client:
        
        if method == "GET":
            result = await client.get(url, headers=headers)
        elif method == "POST":
            result = await client.post(url, json=body, headers=headers)
        # If there's an error print the response
        if result.status_code != 200:
            print(f"Error in request: {result.status_code}: {result.text}")
        
        response = ResponseStatus(result.status_code, None)
        try:
            response.body = result.json()
        except:
            response.body = result.text
        
        if response.status_code == 200:
            response.success = True
        return response