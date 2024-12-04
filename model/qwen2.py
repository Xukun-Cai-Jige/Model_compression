import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, Qwen2VLModel, Qwen2VLConfig, AutoTokenizer, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from model.model import Model
from model.utils import setup_cache_dir

import time

QWEN2_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

def get_model_tokenizer_processor(quantization_mode, model_name):
    """
    :param quantization_mode: 4/8/None. If not None, model would be quanitzed to `quantization_mode` bits using bnb
    :param model_name: Model name or directory path containing the model

    :return: Tuple<Model, Tokenizer, Processor>
    """

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

    config = Qwen2VLConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    cache_dir = setup_cache_dir()

    if quantization_config is not None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


    return model, tokenizer, processor

class Qwen2VL(Model):
    def __init__(self, quantization_mode, from_child=False):

        if not from_child:
            self.model, self.tokenizer, self.processor = get_model_tokenizer_processor(quantization_mode, QWEN2_MODEL_NAME)
        else:
            # Child class is supposed to init all of the following.
            self.model = self.tokenizer = self.processor = None
        
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
    
    def get_model_name(self):
        return f"qwen2_{self.quantization_mode}bit"

    def generate(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        return generated_texts

    def process_image_queries(self, images, queries):
        time_start = time.time()
        messages = []

        for idx, q in enumerate(queries):
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image", "image": images[idx][0], "resized_height": 280,"resized_width": 420},
                        {"type": "text", "text": q["en"]}
                    ]
                }
            ]
            
            messages.append(message)
        
        output = self.process_generate(messages)
        time_end = time.time()

        self.total_processing_time += time_end - time_start
        self.num_processed += 1

        return output


    def process_generate(self, messages):
        self.model.eval()
        
        torch.cuda.empty_cache()
        

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts
    
    def video_inference(self, video_path, user_query, fps=1.0):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{video_path}",
                        "max_pixels": 360 * 420,
                        "fps": fps,
                    },
                    {"type": "text", "text": f"{user_query}"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text


    def get_processor(self):
        return self.processor


class CustomQwen2VL(Qwen2VL):
    def __init__(self, quantization_mode, model, tokenizer, processor):
        super().__init__(quantization_mode, from_child=True)
        
        # Override the model, tokenizer, and processor
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        # Ensure that the model is loaded on the GPU
        # When we patch the model with LowRankModule, it is loaded on CPU,
        # which would trip the inference process.
        self.model.to('cuda')

        if quantization_mode is not None:
            print('WARNING: CustomQwen2VL ignores the quantization mode. Passed value: ' + str(quantization_mode))

    @staticmethod
    def from_low_rank_path(metadata_path, pt_model_path):
        from low_rank.low_rank import patch_model_using_metadata, read_metadata
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN2_MODEL_NAME,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(QWEN2_MODEL_NAME)

        cache_dir = setup_cache_dir()
        config = Qwen2VLConfig.from_pretrained(
            QWEN2_MODEL_NAME,
            trust_remote_code=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN2_MODEL_NAME,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        model = patch_model_using_metadata(model, read_metadata(metadata_path), pt_model_path)

        # Return an instance of CustomQwen2VL
        return CustomQwen2VL(
            quantization_mode=None,
            model=model,
            tokenizer=tokenizer,
            processor=processor
        )
