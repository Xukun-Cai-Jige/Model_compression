import os
import torch
import torch.nn.utils.prune as prune
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

from model.utils import setup_cache_dir

import random
import json

SYSTEM_MESSAGE = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


def format_data(sample):
    image = sample["image"]
    query = sample["query"]
    answer = random.choice(sample["answers"])

    return [
        # {
        #     "role": "system",
        #     "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": 256*28*28,
                    "max_pixels": 256*28*28
                },
                {
                    "type": "text",
                    "text": query,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

# Function to create the dataset
def create_dataset():
    cache_dir = setup_cache_dir()
    train_dataset = load_dataset('nielsr/docvqa_1200_examples', split='train')
    train_dataset = train_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

    eval_dataset = load_dataset('nielsr/docvqa_1200_examples', split='test')
    eval_dataset = eval_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

    return train_dataset, eval_dataset

class MiniDocVQADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        processor = self.processor

        # Apply chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]

         # Process the images to extract inputs
        image_inputs = [process_vision_info(example)[0] for example in examples]
        
        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )

        # Mask padding tokens
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            # Convert image token to ID
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch

def evaluate_model(model):
    from benchmark.docvqa import DocVQA
    from datetime import datetime
    import pandas as pd

    benchmark = DocVQA(model)
    benchmark.evaluate()
    result = benchmark.results()

    print(f"Model: {model.get_model_name()}, Benchmark: DocVQA, Accuracy: {result}")
    now = datetime.now()

    df = pd.DataFrame(columns=["timestamp", "model_name", "benchmark", "accuracy", "memory_utilization", "model_runtime", "additional_results"])
    formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat([df, pd.DataFrame([[formatted_timestamp, model.get_model_name(), 'DocVQA', result, model.get_model_size(), model.get_average_processing_time() , ""]], columns=df.columns)], ignore_index=True)
    df.to_csv("results.csv", index=False)


def prune_model_layers(model, keep_layers):
    """
    Prune transformer layers in the model.
    
    Args:
        model: The transformer model.
        keep_layers: List of layer indices to keep.
        
    Returns:
        The pruned model.
    """
    # Access the model's layers
    transformer_layers = model.model.layers

    # Prune layers
    pruned_layers = torch.nn.ModuleList(
        [transformer_layers[i] for i in keep_layers]
    )
    
    # Replace the model's layers with pruned layers
    model.model.layers = pruned_layers

    return model

def main():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    cache_dir = setup_cache_dir()
    output_dir = os.path.join(os.getcwd(), "qwen2_pruned")

    from model.qwen2 import Qwen2VL, CustomQwen2VL

    qwen2 = Qwen2VL(quantization_mode=None)
    model, tokenizer, processor = qwen2.model, qwen2.tokenizer, qwen2.processor

    # Prune the model
    num_layers = len(model.model.layers)  # Total number of layers
    print(f"Original number of layers: {num_layers}")

    # Actual pruning code would come here
    # ===
    num_layers = len(model.model.layers)  # Total number of layers
    keep_layers = list(range(0, num_layers, 2))  # Keep every alternate layer
    print("Starting structured pruning...")
    model = prune_model_layers(model, keep_layers)
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         prune.ln_structured(module, name='weight', amount=0.2, n=2, dim=0)
    #         prune.remove(module, 'weight')
    print(f"Pruned number of layers: {len(model.model.layers)}")
    # ===

    # Prepare datasets for training

    # dataset_id = "HuggingFaceM4/ChartQA"
    # train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:10%]", "val[:10%]", "test[:10%]"], cache_dir=cache_dir)
    train_dataset = load_dataset('nielsr/docvqa_1200_examples', split='train', cache_dir=cache_dir)
    eval_dataset = load_dataset('nielsr/docvqa_1200_examples', split='test', cache_dir=cache_dir)

    train_dataset = [format_data(example) for example in train_dataset]
    eval_dataset = [format_data(example) for example in eval_dataset]

    # Recover from the latest checkpoint
    def find_latest_checkpoint(output_dir):
        """Find the latest checkpoint in the output directory."""
        checkpoints = [
            os.path.join(output_dir, d) for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if not checkpoints:
            return None
        # Sort by step number and get the latest
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        return latest_checkpoint


    latest_checkpoint = find_latest_checkpoint(output_dir)
    start_step = None
    if latest_checkpoint is not None:
        start_step = int(latest_checkpoint.split("-")[-1]) if latest_checkpoint else 0

    print(f"Resuming training from step {start_step}..." if latest_checkpoint else "Starting training from scratch...")

    training_args = TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=3,
        gradient_checkpointing=True,
        optim='adamw_bnb_8bit',
        learning_rate=4e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=output_dir,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=1,
        dataloader_prefetch_factor=1,
        save_strategy="steps",
        save_steps=200,  # Save checkpoint every 200 steps
        save_total_limit=2,  # Keep only the last 2 checkpoints to save space
    )
    os.makedirs(output_dir, exist_ok=True)

    data_collator = MiniDocVQADataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Fine-tune the model
    print("\nStarting fine-tuning...\n")
    trainer.train()
    print("\nFine-tuning completed.")


    print("Saving model with device_map='auto' for offloading...")
    model.save_pretrained(output_dir, safe_serialization=False, max_shard_size="500MB", device_map="auto")
    print("Model saved successfully.")

    print("Saving processor")
    if not hasattr(processor, 'chat_template'):
        processor.chat_template = None

    print("Saving processor...")
    processor.save_pretrained(output_dir)
    print(f"Processor saved to {output_dir}")

    custom_model = CustomQwen2VL(None, model, tokenizer, processor)
    # Or custom_model could be loaded as following:
    evaluate_model(custom_model)
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()