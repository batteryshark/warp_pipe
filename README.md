![alt text](./images/wplogo.png "Warp Pipe Logo")

# Warp Pipe - A local multi-LLM Proxy Service

After lots of screwing around with various LLM providers and local servers, I wanted something that would make switching and interop a little easier. This has been a few iterations in the making but I think it's worth releasing while I continue to make updates.

## Why? And why make them all like OpenAI?

Why not? Why not try to make various endpoints all act the same? The OpenAI API is pretty widely accepted with everyone else's projects. It seems kinda fitting to model that with the others.

## Don't providers already have OpenAI endpoints?

Some do... with varying levels of compatibility. Some have function calling, some don't. Some support JSON, some don't... some let you have far more control such as controlling the context limit or offloading... and some don't.

For instance, Ollama's API lets you load a model but if the model file doesn't specify num_ctx, you get 2048 and no way to increase it.

## Features

* Per-Provider Configuration: Adding presets and aliases by provider allows you to modify what models the various adapters serve and how they get served.
* Crude API Authorization: For when you don't want to expose an llm proxy without some kind of token.
* Shiny Uvicorn/FastAPI Backend: Because I wanted an alternative to Flask
* Streaming Mode Support: Where some providers don't support this out of the box.
* n Generations: Because again, not everyone supports this with their API.
* Base64 Embeddings: Because it's a pretty simple add to bring embeddings endpoints to parity.
* [Experimental] Function Calling: For providers that don't support function calling yet (e.g. Ollama, most of the Mistral models)
* Additional Configuration Headers:
    - LLM_PROVIDER: Specify the provider you want (optional, a default is set in the config)
    - PROVIDER_AUTH: Bring your own api key (in case you don't want to globally set one)
    - MAX_CONTEXT: For local llms, specify a context window limit.



## Compatibility List

### Mistral

* [x] List Models
* [x] Get Model Info
* [x] Embeddings
* [x] Chat Completions
* [x] JSON Mode
* [x] Function Calling (works on mistral-tiny, their API blocks function calling that doesn't go to mistral-large)

### Groq

* [x] List Models
* [x] Get Model Info
* [ ] Embeddings (No Embedding Models, workaround needed)
* [x] Chat Completions
* [x] JSON Mode
* [x] Function Calling (works on mixtral)

### OpenAI

* [x] List Models
* [x] Get Model Info
* [x] Embeddings
* [x] Chat Completions
* [-] JSON Mode (Not all models, workaround needed)
* [-] Function Calling (Not all models, workaround needed)

### Ollama

* [x] List Models
* [x] Get Model Info
* [x] Embeddings
* [x] Chat Completions

* [x] Conversion from Image URL to Base64 Images for Multimodal
* [x] Function Calling (Tested with Mistral Instruct Q5_K_M)


### Anthropic:
* [x] List Models
* [x] Get Model Info
* [x] Embeddings (No Embedding Models, workaround needed)
* [x] JSON Mode (Pretty Hacky)
* [x] Chat Completions
* [x] Function Calling


## Developer Notes

Lets-a-Go!