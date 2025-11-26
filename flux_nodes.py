import os
import torch
import numpy as np
from PIL import Image
import io
import replicate
import requests
import json # Aggiunto per il debug JSON

# Defines the current directory to look for the key file
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class ReplicateAuthNode:
    """
    Reads the Replicate API Key from a local text file.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "load_key"
    CATEGORY = "FluxReplicate"

    def load_key(self):
        key_file = os.path.join(CURRENT_DIR, "replicate_api_key.txt")
        
        if not os.path.exists(key_file):
            raise FileNotFoundError(f"Error: The file {key_file} was not found. Please create a 'replicate_api_key.txt' file with your key inside this folder.")
            
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
            
        if not api_key:
            raise ValueError("Error: The replicate_api_key.txt file is empty.")
            
        # Mask the key in logs for security
        print(f"FluxReplicate: Key loaded (starts with {api_key[:4]}...)")
        
        return (api_key,)

class Flux2ProGenerator:
    """
    Main node to generate images with Flux 2 Pro on Replicate.
    Includes JSON payload debug output.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Define optional inputs for images dynamically
        optional_images = {}
        for i in range(1, 11):
            optional_images[f"image_{i}"] = ("IMAGE",)

        return {
            "required": {
                "api_key": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": ([
                    "custom", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"
                ], {"default": "1:1"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}), 
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "resolution": (["4 MP", "2 MP", "1 MP", "0.25 MP"], {"default": "1 MP"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "safety_tolerance": ("INT", {"default": 2, "min": 1, "max": 5}),
                "output_quality": ("INT", {"default": 80, "min": 1, "max": 100}),
            },
            "optional": optional_images
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "DEBUG_PAYLOAD")
    FUNCTION = "generate_flux"
    CATEGORY = "FluxReplicate"

    def generate_flux(self, api_key, prompt, aspect_ratio, width, height, resolution, seed, safety_tolerance, output_quality, **kwargs):
        
        # 1. Setup Replicate Client
        os.environ["REPLICATE_API_TOKEN"] = api_key
        
        # 2. Process Input Images (Guidance)
        input_images_files = []
        
        # Iterate over optional arguments (image_1 ... image_10)
        for i in range(1, 11):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                tensor_img = kwargs[key]
                
                # Conversion Tensor ComfyUI -> PIL Image -> BytesIO
                tensor_img = tensor_img[0] 
                
                # Convert from float32 (0-1) to uint8 (0-255)
                i_img = 255. * tensor_img.cpu().numpy()
                img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
                
                # Save to memory buffer
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                input_images_files.append(img_byte_arr)

        # 3. Prepare Payload
        input_payload = {
            "prompt": prompt,
            "output_format": "png",
            "output_quality": output_quality,
            "safety_tolerance": safety_tolerance,
            "seed": seed
        }

        # Logic for Aspect Ratio vs Custom Width/Height
        if aspect_ratio == "custom":
            print(f"FluxReplicate: Using custom dimensions {width}x{height}")
            input_payload["aspect_ratio"] = "custom"
            input_payload["width"] = width
            input_payload["height"] = height
            # 'resolution' is ignored when using custom dimensions to avoid conflicts
        else:
            print(f"FluxReplicate: Using aspect ratio {aspect_ratio} at {resolution}")
            input_payload["aspect_ratio"] = aspect_ratio
            input_payload["resolution"] = resolution

        # Add images only if present
        if input_images_files:
            input_payload["input_images"] = input_images_files

        # DEBUG: Generate JSON string of payload
        # Note: We cannot serialise the BytesIO objects in the JSON debug string easily, 
        # so we put a placeholder text for images in the debug print
        debug_payload = input_payload.copy()
        if "input_images" in debug_payload:
            debug_payload["input_images"] = [f"<Image Data {i+1}>" for i in range(len(input_images_files))]
        
        payload_json_str = json.dumps(debug_payload, indent=4)
        print(f"FluxReplicate: Sending request... Payload:\n{payload_json_str}")

        # 4. API Call
        try:
            model_id = "black-forest-labs/flux-2-pro" 
            
            output = replicate.run(
                model_id,
                input=input_payload
            )
            
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = str(output)

            print(f"FluxReplicate: Image generated: {image_url}")

        except Exception as e:
            raise RuntimeError(f"Error during generation with Replicate: {str(e)}")

        # 5. Download and Convert Output -> ComfyUI Tensor
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,] 
            
            return (img_tensor, payload_json_str)

        except Exception as e:
             raise RuntimeError(f"Error during image download or conversion: {str(e)}")

# Class Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ReplicateAuthNode": ReplicateAuthNode,
    "Flux2ProGenerator": Flux2ProGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateAuthNode": "Flux Replicate Auth (File)",
    "Flux2ProGenerator": "Flux 2 Pro Generator"
}