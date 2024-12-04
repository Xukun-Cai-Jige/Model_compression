import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, Qwen2VLModel, Qwen2VLConfig, AutoTokenizer, Qwen2VLForConditionalGeneration,AutoModelForCausalLM
from model.model import Model
from model.utils import setup_cache_dir, extract_frames_video, read_video_pyav
import PIL
import av

import numpy as np

import time

def get_model_tokenizer_processor(quantization_mode):
    model_name = "microsoft/Phi-3.5-vision-instruct"

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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            _attn_implementation='eager'    
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            _attn_implementation='eager'    
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, num_crops= 4)

    return model, tokenizer, processor

class Phi3_5(Model):
    def __init__(self, quantization_mode):
        self.model, self.tokenizer, self.processor = get_model_tokenizer_processor(quantization_mode)

        self.quantization_mode = quantization_mode
        self.num_processed = 0
        self.total_processing_time = 0

    def get_average_processing_time(self):
        if self.num_processed == 0:
            return 0
        return self.total_processing_time

    def process(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs

    def generate(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        return generated_texts

    def get_model_name(self):
        return f"phi3_5_{self.quantization_mode}bit"

    def process_image_queries(self, images, queries):
        self.model.eval()
        
        torch.cuda.empty_cache()

        time_start = time.time()

        images = images[0][0]
        
        texts = []
        placeholder = f"<|image_1|>\n"
        for idx, q in enumerate(queries):
            if idx > 0:
                print("Multiple Queries not supported")
            print(q["en"])
            if type(q["en"]) == list:
                q["en"] = q["en"][0]
            messages = [
            {   
                "role": "user", 
                "content": "Answer briefly." + placeholder+ " " + q["en"]},
            ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0") 
        generation_args = { 
            "max_new_tokens": 128, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        # inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        # inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)

        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        time_end = time.time()

        self.total_processing_time += time_end - time_start
        self.num_processed += 1
        
        return generated_texts

    def video_inference(self, video_path, user_query, fps=1.0, num_samples=8):
        self.model.eval()
        torch.cuda.empty_cache()

        output_dir = "phi3_frames"
        
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / num_samples).astype(int)

        clip = read_video_pyav(container, indices, output_dir, True)
        num_processed = len(clip)

        print("Processed {} frames".format(num_processed))

        images = []
        placeholder = ""

        for i in range(1,num_processed+1):
            image_path = f"{output_dir}/frame_{i-1}.jpg"
            print("Processing image: ", image_path)
            image = PIL.Image.open(image_path)
            images.append(image)
            placeholder += f"<|image_{i}|>\n"
        
        messages = [
            {"role": "user", "content": placeholder+"Summarize the following video."},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0") 
        generation_args = { 
            "max_new_tokens": 128, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        # inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        # inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)

        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        
        return generated_texts

    def process_generate(self, original_texts, images):
        pass

    def get_processor(self):
        return self.processor