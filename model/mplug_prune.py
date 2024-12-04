import torch
import os 
import sys

#from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer, BitsAndBytesConfig,  AutoProcessor
from model.model import Model
from model.utils import setup_cache_dir
mplug_owl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mPLUG-Owl'))
sys.path.append(mplug_owl_path)
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from PIL import Image

import time

def save_model_in_chunks(model, save_path):
    """
    Save model in smaller chunks to avoid memory exhaustion.
    Args:
        model (transformers.PreTrainedModel): The model instance.
        save_path (str): Path to save the model.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save configuration
    model.config.save_pretrained(save_path)
    
    # Save tokenizer
    if hasattr(model, "tokenizer"):
        model.tokenizer.save_pretrained(save_path)
    
    # Save the state dictionary in chunks
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        torch.save(value, os.path.join(save_path, f"{key}.pt"))
    print(f"Model saved in chunks at {save_path}")
import psutil

def log_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")
def prune_language_model_layers(model, save_path):
    """
    Prunes every other layer in the language model and saves the pruned model.
    Args:
        model (transformers.PreTrainedModel): The loaded model instance.
        save_path (str): The directory to save the pruned model.
    """
    language_model = model.language_model.model
    original_layers = language_model.layers
    pruned_layers = torch.nn.ModuleList([layer for i, layer in enumerate(original_layers) if i % 2 == 0])
    language_model.layers = pruned_layers
    model.config.num_hidden_layers = len(pruned_layers)

    # Log memory usage before saving
    log_memory()

    # Save the pruned model in chunks to avoid memory issues
    save_model_in_chunks(model, save_path)

    # Log memory usage after saving
    log_memory()
def get_model_tokenizer_processor(quantization_mode):
    model_name = "MAGAer13/mplug-owl-llama-7b"

    if quantization_mode == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantization_mode == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        # load the default model 
        quantization_config = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    cache_dir = setup_cache_dir()
    
    if quantization_config is not None:
        model = MplugOwlForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    else:
        model = MplugOwlForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
        )
    save_path = os.path.join(os.getcwd(), "mplug_pruned")
    prune_language_model_layers(model, save_path)
    image_processor = MplugOwlImageProcessor.from_pretrained(model_name)
    tokenizer = MplugOwlTokenizer.from_pretrained(model_name)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    return model, tokenizer, processor

class Mplug(Model):
    def __init__(self, quantization_mode):
        self.model, self.tokenizer, self.processor = get_model_tokenizer_processor(quantization_mode)
        
        self.quantization_mode = quantization_mode
        self.num_processed = 0
        self.total_processing_time = 0

    def get_average_processing_time(self):
        if self.num_processed == 0:
            return 0
        return self.total_processing_time / self.num_processed
    
    def get_model_name(self):
        return f"Mplug_{self.quantization_mode}bit"
    
    def generate(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        return generated_texts
    
    def process_image_queries(self, images, queries):
        torch.cuda.empty_cache()
        start_time = time.time()
        
        q = queries[0]
        prompts = [
            f'''The following is a conversation between a curious human and AI assistant. The assistant gives a direct answer in maximum two words to the user's question.
            Human: <image>
            Human: {query}
            AI: ''' for query in queries
        ]

        # The image paths should be placed in the image_list and kept in the same order as in the prompts.
        # We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }
        images = [img for sublist in images for img in sublist] if any(isinstance(i, list) for i in images) else images
        images = [Image.open(img) if isinstance(img, str) else img for img in images]
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        # sentences =  self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        sentences = self.tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        end_time = time.time()
        self.total_processing_time += end_time - start_time
        self.num_processed += 1

        # print(sentence)
        # for query, sentence in zip(queries, sentences):
        #     print(f"Query: {query}")
        #     print(f"Response: {sentence}")
        return sentences



    def get_processor(self):
        return self.processor