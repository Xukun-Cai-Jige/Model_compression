import torch
import av
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaProcessor, VideoLlavaForConditionalGeneration, AutoTokenizer
from model.model import Model
from model.utils import setup_cache_dir,read_video_pyav
from huggingface_hub import hf_hub_download
import time

import numpy as np

def get_model_tokenizer_processor(quantization_mode):
    model_name = "LanguageBind/Video-LLaVA-7B-hf"

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
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    else:
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="sequential",
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, tokenizer, processor

class VideoLLava(Model):
    def __init__(self, quantization_mode):
        self.model, self.tokenizer, self.processor = get_model_tokenizer_processor(quantization_mode)
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        
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
        return f"video_llava_{self.quantization_mode}bit"

    def process_image_queries(self, images, queries):
        start_time = time.time()
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        image = images[0][0]
        assert len(images) == 1, "Only one image is supported"
        assert len(queries) == 1, "Only one query is supported"
        q = queries[0]
        prompt = [
            f"USER: Answer briefly. <image> {q['en']} ASSISTANT:"
        ]

        inputs = self.processor(text=prompt, images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generate_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_texts = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_texts = [g.split("ASSISTANT:")[1].strip() for g in generated_texts]
        end_time = time.time()

        self.total_processing_time += (end_time - start_time)
        self.num_processed += 1

        return generated_texts


    def video_inference(self, video_path, user_query, fps=1.0):
        video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)

        clip = read_video_pyav(container=container, indices=indices)
        prompt = f"USER: <video>\ {user_query} ASSISTANT:"

        inputs = self.processor(text=prompt, videos=clip, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        generate_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_texts = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_texts = [g.split("ASSISTANT:")[1].strip() for g in generated_texts]

        return generated_texts


    def process_generate(self, original_texts, images):
        pass

    def get_processor(self):
        return self.processor