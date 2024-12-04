import torch
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
import os

def setup_cache_dir():
    """Setup local cache directories for model and data"""
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set HuggingFace cache directories
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
    
    return cache_dir

cache_dir = setup_cache_dir()

model_name = "microsoft/Phi-3.5-vision-instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,  # Cache the model here
    ).to("cuda")

for name, param in model.named_parameters():
        print(f"Layer: {name}, Size: {param.size()}")