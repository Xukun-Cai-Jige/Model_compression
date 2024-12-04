from datasets import load_dataset
from tqdm import tqdm
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, Qwen2VLModel, Qwen2VLConfig
from qwen_vl_utils import process_vision_info
from model.model import Model
from benchmark.benchmark import Benchmark
from model.utils import setup_cache_dir
import base64
from io import BytesIO
from benchmark.utils import average_normalized_levenshtein_similarity, average_bert_score_f1_value
from PIL import Image
import os

class DocVQA_DEMO(Benchmark):
    def __init__(self, model: Model):
        self.model = model
        self.processor = model.get_processor()
        self.model_type = self.model.get_model_name()
        self.cache_dir = os.path.join(setup_cache_dir(), "datasets")
        self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="test", cache_dir=self.cache_dir)

        # Load only the first `num_examples` examples
        self.dataset = self.dataset.select(range(min(10, len(self.dataset))))
        self.dataset = self.dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

        self.answers_unique = []
        self.generated_texts_unique = []
        self.output_file = "docvqa_output.txt"  # File to save the output

    def evaluate(self):
        EVAL_BATCH_SIZE = 1

        with open(self.output_file, "w") as f:
            for i in tqdm(range(0, len(self.dataset), EVAL_BATCH_SIZE)):
                examples = self.dataset[i: i + EVAL_BATCH_SIZE]
                self.answers_unique.extend(examples["answers"])
                images = [[im] for im in examples["image"]]
                queries = examples["query"]

                # Print content and save to the file
                print(f"Test Case {i + 1}:", file=f)
                print(f"Query: {queries[0]}", file=f)
                print(f"Expected Answer: {examples['answers'][0]}", file=f)

                img = images[0][0]
                img_path = f"docvqa_image_{i + 1}.png"
                img.save(img_path)
                print(f"Image saved as {img_path}")

                output = self.model.process_image_queries(images, queries)
                print(f"Model Output: {output[0]}", file=f)
                print("-" * 50, file=f)

                self.generated_texts_unique.extend(output)

    def results(self):
        self.generated_texts_unique = [g.strip().strip(".") for g in self.generated_texts_unique]
        anls = average_normalized_levenshtein_similarity(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )

        bert_score_f1 = average_bert_score_f1_value(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )

        return anls, bert_score_f1