import torch

#from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer, BitsAndBytesConfig,  AutoProcessor
from model.model import Model
from model.utils import setup_cache_dir
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

import time

def get_model_tokenizer_processor(quantization_mode):
    model_name = "MAGAer13/mplug-owl-llama-7b-video"

    if quantization_mode == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
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
            f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {q}
            AI: '''
        ]

        # The image paths should be placed in the image_list and kept in the same order as in the prompts.
        # We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        sentence =  self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        
        end_time = time.time()
        self.total_processing_time += end_time - start_time
        self.num_processed += 1

        print(sentence)
        return sentence



    def get_processor(self):
        return self.processor