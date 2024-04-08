from datetime import datetime, timezone
import time
import struct
import base64
import httpx

def convert_datetime_to_epoch(datetime_str):
    """Convert a datetime string to epoch time."""
    datetime_obj = datetime.fromisoformat(datetime_str)
    epoch_time = datetime_obj.replace(tzinfo=timezone.utc).timestamp()
    return int(epoch_time)

def encode_embeddings_to_base64(embeddings):
    # Step 1: Convert the list of floats into binary data.
    # Using 'f' as format specifier for single precision float.
    # Use 'd' instead of 'f' for double precision.
    binary_data = struct.pack(f'{len(embeddings)}f', *embeddings)
    
    # Step 2: Encode this binary data into a base64 byte string.
    base64_bytes = base64.b64encode(binary_data)
    
    # Convert the byte string to a regular string for easier use
    base64_string = base64_bytes.decode('utf-8')
    
    return base64_string

async def download_image_from_url_and_encode_b64(image_url):
    # Add an indented block of code here
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        image_bytes = response.content
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image
    return ""
    