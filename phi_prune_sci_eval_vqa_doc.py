import os
import time
import torch
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, AutoProcessor, AutoModelForCausalLM
from benchmark.utils import average_bert_score_f1_value
import random
# Constants
TOTAL_DATA_POINTS = 1200
TRAIN_TEST_SPLIT = 0.8
TRAIN_SIZE = int(TOTAL_DATA_POINTS * TRAIN_TEST_SPLIT)
TEST_SIZE = TOTAL_DATA_POINTS - TRAIN_SIZE

# Function to load and split ScienceQA dataset
def setup_cache_dir():
    """Setup local cache directories for model and data"""
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set HuggingFace cache directories
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
    
    return cache_dir

def is_valid_example(example):
    """Check if an example is valid."""
    # Example must have an image and at least one non-empty choice
    return (
        example.get("image") is not None
    )
# Load and split ScienceQA dataset
def load_and_split_scienceqa():
    """Load the ScienceQA dataset, filter valid examples, and split into train/test subsets."""
    print("Loading ScienceQA dataset...")
    dataset = load_dataset("derek-thomas/ScienceQA", split="test", cache_dir=setup_cache_dir())
    
    # Filter valid examples
    print("Filtering valid examples...")
    valid_dataset = dataset.filter(is_valid_example)

    # Select the first 2000 valid examples
    print(f"Total valid examples: {len(valid_dataset)}")
    valid_dataset = valid_dataset.select(range(min(TOTAL_DATA_POINTS, len(valid_dataset))))

    # Shuffle and split into train and test sets
    print("Shuffling and splitting dataset...")
    valid_dataset = valid_dataset.shuffle(seed=21)
    train_dataset = valid_dataset.select(range(TRAIN_SIZE))
    test_dataset = valid_dataset.select(range(TRAIN_SIZE, len(valid_dataset)))

    return DatasetDict({"train": train_dataset, "test": test_dataset})

# Load and split VQA v2 dataset
def load_and_split_vqa2():
    """Load the VQA v2 dataset and split it into train and test subsets."""
    print("Loading VQA v2 dataset...")
    dataset = load_dataset("merve/vqav2-small", split="validation", cache_dir=setup_cache_dir())
    dataset = dataset.select(range(TOTAL_DATA_POINTS))  # Select first 2000 data points

    # Shuffle and split
    dataset = dataset.shuffle(seed=21)
    train_dataset = dataset.select(range(TRAIN_SIZE))
    test_dataset = dataset.select(range(TRAIN_SIZE, TOTAL_DATA_POINTS))

    return DatasetDict({"train": train_dataset, "test": test_dataset})

# Load and split docvqa dataset
def load_and_split_docvqa():
    """Load the DocVQA dataset and split it into train and test subsets."""
    print("Loading DocVQA dataset...")
    dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", cache_dir=setup_cache_dir())

    # Select first TOTAL_DATA_POINTS
    dataset = dataset.select(range(min(TOTAL_DATA_POINTS, len(dataset)))) 

    # Shuffle and split into train and test sets
    print("Shuffling and splitting dataset...")
    dataset = dataset.shuffle(seed=21)
    train_dataset = dataset.select(range(TRAIN_SIZE))
    test_dataset = dataset.select(range(TRAIN_SIZE, len(dataset)))

    return DatasetDict({"train": train_dataset, "test": test_dataset})

#datacollator for docvqa
class MiniDocVQADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Batch size must be 1 for Phi-3-V models'
        example = examples[0]

        image = example['image']
        question = example['query']['en']
        answer = random.choice(example['answers'])
        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}\nAnswer briefly.',
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{answer}<|end|>\n<|endoftext|>'

        # Process input and labels
        batch = self.processor(prompt, [image], return_tensors='pt')
        prompt_input_ids = batch['input_ids']
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)

        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels
        return batch

# Custom Data Collator for VQA v2
class VQA2DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Batch size must be 1 for Phi-3-V models'
        images = [example["image"] for example in examples]
        questions = examples["question"]
        answer = examples["multiple_choice_answer"]


        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{questions[0]}\nAnswer briefly.',
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{answer}<|end|>\n<|endoftext|>'


        batch = self.processor(prompt, [images], return_tensors="pt", padding=True, truncation=True)
        prompt_input_ids = batch['input_ids']
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
        # Set ignore index for prompts
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels
        return batch

# Custom Data Collator for ScienceQA
class ScienceQACollator:
    def __init__(self, processor):
        self.processor = processor

    def _format_question(self, example):
        """Format question with choices and additional context."""
        question_text = example.get("question", [""])[0]  # Extract the first   question if it's a list
        choices = example.get("choices", [[]])[0]  # Extract the first set of   choices
        hint = example.get("hint", None)  # Get the hint, if available

        # Ensure `hint` is valid and non-empty
        if isinstance(hint, list) and len(hint) > 0:
            hint = hint[0]
        else:
            hint = ""  # Default to an empty string if no valid hint is present

        # Build the question text
        formatted_question = f"Question: {question_text}\n\n"
        formatted_question += "Choices:\n"
        for i, choice in enumerate(choices):
            formatted_question += f"{chr(ord('A') + i)}. {choice}\n"

        # Include the hint if available
        if hint:
            formatted_question += f"\nHint: {hint}\n"

        formatted_question += "\nPlease answer directly with only the letter of the     correct option."
        return formatted_question

    def __call__(self, examples):
        assert len(examples) == 1, 'Batch size must be 1 for Phi-3-V models'
        #print(examples[0])
        images = [example["image"] for example in examples]
        questions = self._format_question(examples[0])
        answer = examples[0]["answer"]


        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{questions[0]}\n answer in one word.',
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = chr(ord('A') + int(examples[0]["answer"]))


        batch = self.processor(prompt, images, return_tensors="pt", padding=True, truncation=True)
        prompt_input_ids = batch['input_ids']
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
        # Set ignore index for prompts
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels
        return batch


#Docvqa
class DocVQAEvaluator:
    def __init__(self, model, processor, test_dataset):
        self.model = model
        self.processor = processor
        self.test_dataset = test_dataset
        self.answers_unique = []
        self.generated_texts_unique = []

    def results(self):
        # Flatten and clean up the generated texts
        self.generated_texts_unique = [
            str(g[0]).strip().strip(".") if isinstance(g, list) and len(g) > 0 else str(g).strip().strip(".")
            for g in self.generated_texts_unique
        ]

        # Calculate BERTScore
        bert_score_f1 = average_bert_score_f1_value(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )

        print(f"BERT Score F1: {bert_score_f1}")
        return bert_score_f1

    def format_query(self, question):
        """
        Format the query for the model.

        Args:
            question (str): The question text from the dataset.

        Returns:
            dict: Formatted query in a dictionary structure.
        """
        formatted_query = {
            "role": "user",
            "content": f"<|image_1|>\n{question}\nAnswer briefly.",
        }
        return formatted_query

    def process_query(self, image, query):
        """
        Process a single query using the model.

        Args:
            image (PIL.Image.Image or similar): The input image.
            query (str): The input question text.

        Returns:
            str: The model's generated answer.
        """
        self.model.eval()
        torch.cuda.empty_cache()

        # Format the query
        formatted_query = self.format_query(query)

        # Tokenize the input
        prompt = self.processor.tokenizer.apply_chat_template(
            [formatted_query], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")

        # Generate the output
        generation_args = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "do_sample": False,
        }
        generated_ids = self.model.generate(
            **inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args
        )
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        """
        print("Evaluating model on test dataset...")
        for example in tqdm(self.test_dataset):
            # Extract image and question from the dataset example
            image = example["image"]
            question = example["query"]["en"]
            ground_truth = random.choice(example["answers"])  # Randomly pick one ground truth answer

            # Process the query and store results
            generated_answer = self.process_query(image, question)
            self.generated_texts_unique.append(generated_answer)
            self.answers_unique.append(ground_truth)

        print("Evaluation completed.")
# ScienceQA
class ScienceQAEvaluator:
    def __init__(self, model, processor, test_dataset):
        self.model = model
        self.processor = processor
        self.test_dataset = test_dataset
        self.answers_unique = []
        self.generated_texts_unique = []
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
                "content": "Answer in one word." + placeholder+ " " + q["en"]},
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
    
    def _format_question(self, example):
        """Format question with choices and additional context."""
        question_text = example.get("question", [""])[0]  # Extract the first question if it's a list
        choices = example.get("choices", [[]])[0]  # Extract the first set of choices
        hint = example.get("hint", [""])[0]  # Use the first hint, if available

        # Build the question text
        formatted_question = f"Question: {question_text}\n\n"
        formatted_question += "Choices:\n"
        for i, choice in enumerate(choices):
            formatted_question += f"{chr(ord('A') + i)}. {choice}\n"

        # Include the hint if available
        if hint:
            formatted_question += f"\nHint: {hint}\n"

        formatted_question += "\nPlease answer directly with only the letter of the correct option."
        return formatted_question
    def evaluate(self):
        print("Evaluating model on test dataset...")
        for example in tqdm(self.test_dataset):
            images = [example["image"]]
            questions = [self._format_question(example)]
            self.answers_unique.append(chr(ord('A') + int(example["answer"][0])))
            generated_text = self.process_image_queries(images, questions)
            self.generated_texts_unique.append(generated_text)

    def results(self):
        self.generated_texts_unique = [g.strip().strip(".") for g in self.generated_texts_unique]

        # Calculate BERTScore
        bert_score_f1 = average_bert_score_f1_value(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )

        print(f"BERT Score F1: {bert_score_f1}")
        return bert_score_f1


# Evaluation class for VQA v2
class VQA_v2_Evaluator:
    def __init__(self, model, processor, test_dataset):
        self.model = model
        self.processor = processor
        self.test_dataset = test_dataset
        self.answers_unique = []
        self.generated_texts_unique = []
    def process_image_queries(self, images, queries):
        self.model.eval()
        
        torch.cuda.empty_cache()

        time_start = time.time()

        images = images[0]
        
        texts = []
        placeholder = f"<|image_1|>\n"

        messages = [
        {   
           "role": "user", 
            "content": "Answer briefly." + placeholder+ " " + queries[0]},
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
        
        return generated_texts
    def evaluate(self):
        print("Evaluating model on test dataset...")
        for example in tqdm(self.test_dataset):
            images = [example["image"]]
            question = [example["question"]]
            self.answers_unique.append(example["multiple_choice_answer"])
            generated_text = self.process_image_queries(images, question)
            self.generated_texts_unique.append(generated_text)

    def results(self):
        #print(self.generated_texts_unique)

        # Flatten the nested lists if needed and convert to strings
        flattened_generated_texts = [
            str(item[0]).strip().strip(".") if isinstance(item, list) and len(item)     > 0 else str(item).strip().strip(".")
            for item in self.generated_texts_unique
        ]

        self.generated_texts_unique = flattened_generated_texts

        # Calculate BERTScore
        bert_score_f1 = average_bert_score_f1_value(
            ground_truth=self.answers_unique, predicted_answers=self.   generated_texts_unique,
        )

        print(f"BERT Score F1: {bert_score_f1}")
        return bert_score_f1


# Main function
def main():
    model_name = "microsoft/Phi-3.5-vision-instruct"
    output_dir = os.path.join(os.getcwd(), "vqa2_phi3")
    cache_dir = setup_cache_dir()

    # Load and split dataset
    scienceqa_datasets = load_and_split_scienceqa()
    vqa2_datasets = load_and_split_vqa2()
    docvqa_datasets = load_and_split_docvqa()

    train_dataset = scienceqa_datasets["train"]
    vqa2_test_dataset = vqa2_datasets["test"]
    docvqa_test_datasets = docvqa_datasets["test"]

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to("cuda")

    # Define training arguments
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        optim='adamw_torch',
        learning_rate=4e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=output_dir,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_strategy="no",
    )

    # Custom Data Collator
    data_collator = ScienceQACollator(processor)

    # Fine-tuning with Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("\nStarting fine-tuning...")
    start_time = time.time()
    trainer.train()
    fine_tuning_time = time.time() - start_time
    print(f"Fine-tuning completed in {fine_tuning_time:.2f} seconds.")

    # Save model and processor
    model.save_pretrained(output_dir, safe_serialization=False, max_shard_size="500MB", device_map="auto")
    if not hasattr(processor, 'chat_template'):
        processor.chat_template = None
    processor.save_pretrained(output_dir)

    # Evaluate
    evaluator = VQA_v2_Evaluator(model, processor,vqa2_test_dataset)
    start_time = time.time()
    evaluator.evaluate()
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds.")

    # Results
    evaluator.results()
    print(f"Fine-tuning time: {fine_tuning_time:.2f} seconds, Evaluation time: {evaluation_time:.2f} seconds.")


if __name__ == "__main__":
    main()
