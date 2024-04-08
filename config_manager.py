import os
import json
import uuid

APP_CONFIG = {}
CONFIG_LOADED = False
CONFIG_PATH = os.environ.get("WARP_PIPE_CONFIG_PATH", None)
if CONFIG_PATH is None:
    CONFIG_PATH = os.path.expanduser("~/.warp_pipe.conf")

CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
                    "api_keys": [str(uuid.uuid4())],
                    "auth_enforcement_enabled": False,
                    "host": "localhost",
                    "port": 32823,
                    "allowed_origins": ["localhost"],
                    "default_provider": "OLLAMA",
                    "provider_options": {}
}

def save_config(config):
    with open(CONFIG_PATH, "w") as config_file:
        json.dump(config, config_file, indent=4)

def load_config():
    global APP_CONFIG
    global CONFIG_LOADED
    try:
        with open(CONFIG_PATH, "r") as config_file:
            config_data = json.load(config_file)
    except:
        config_data = DEFAULT_CONFIG
        save_config(config_data)
    CONFIG_LOADED = True
    APP_CONFIG["api_keys"] = config_data.get("api_keys",[])
    APP_CONFIG["auth_enforcement_enabled"] = config_data.get("auth_enforcement_enabled", False)
    APP_CONFIG["host"] = config_data.get("host", "localhost")
    APP_CONFIG["port"] = config_data.get("port", 32823)
    APP_CONFIG["allowed_origins"] = config_data.get("allowed_origins", ["localhost"])
    APP_CONFIG["default_provider"] = config_data.get("default_provider", "OLLAMA")
    APP_CONFIG["provider_options"] = config_data.get("provider_options",{})

def get_config():
    global CONFIG_LOADED
    global APP_CONFIG
    if not CONFIG_LOADED:
        load_config()
    return APP_CONFIG

def get_provider_options(provider, default_options={}):
    config = get_config()
    provider_options = config["provider_options"].get(provider)
    if provider_options is None:
        provider_options = default_options
        set_provider_options(provider, provider_options)
    return provider_options

def set_provider_options(provider, options):
    config = get_config()
    config["provider_options"][provider] = options
    save_config(APP_CONFIG)

def init_config():
    global APP_CONFIG
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        print(f"Configuration file not found at {CONFIG_PATH}. A new configuration file has been created with default values. Please edit this file to configure Warp Pipe.")
    load_config()

