import torch
import av
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, Qwen2VLModel, Qwen2VLConfig, AutoTokenizer, Qwen2VLForConditionalGeneration
from model.model import Model
from model.utils import setup_cache_dir, read_video_pyav
from huggingface_hub import hf_hub_download
import PIL

import time
import numpy as np

def get_model_tokenizer_processor(quantization_mode):
    model_name = "HuggingFaceM4/idefics2-8b"

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
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, tokenizer, processor

class Idefics2(Model):
    def __init__(self, quantization_mode):
        self.model, self.tokenizer, self.processor = get_model_tokenizer_processor(quantization_mode)
        
        self.quantization_mode = quantization_mode
        self.num_processed = 0
        self.total_processing_time = 0

    def get_average_processing_time(self):
        if self.num_processed == 0:
            return 0
        return self.total_processing_time / self.num_processed

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
        return f"idefics2_{self.quantization_mode}bit"

    def process_image_queries(self, images, queries):
        start_time = time.time()
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        texts = []
        for q in queries:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": q["en"]}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)

        end_time = time.time()
        self.total_processing_time += end_time - start_time
        self.num_processed += 1

        return generated_texts

    def video_inference(self, video_path, user_query, fps=1.0, num_samples=8):
        self.model.eval()
        
        torch.cuda.empty_cache()

        output_dir = "idefics2_frames"
        
        container = av.open(video_path)
        
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)

        clip = read_video_pyav(container, indices, output_dir, True)
        num_processed = len(clip)

        
        texts = []
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Given {num_samples} uniformly sampled frames from a video. Answer the user query: {user_query}"},
                ]
            }
        ]

        images = []
        for i in range(0,num_processed):
            image_path = f"{output_dir}/frame_{i}.jpg"
            image = PIL.Image.open(image_path)
            images.append(image)
            messages[0]["content"].append({"type": "image"})

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        texts.append(text.strip())

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)

        return generated_texts


    def process_generate(self, original_texts, images):
        pass

    def get_processor(self):
        return self.processor