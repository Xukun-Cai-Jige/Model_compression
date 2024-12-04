import os
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

from model.utils import setup_cache_dir

import random
import time
import json

def get_skip_layers():
    return ['model.layers.20.mlp.gate_proj',
    'model.layers.20.mlp.up_proj',
    'model.layers.20.mlp.down_proj',
    'model.layers.21.self_attn.q_proj',
    'model.layers.21.self_attn.k_proj',
    'model.layers.21.self_attn.v_proj',
    'model.layers.21.self_attn.o_proj',
    'model.layers.21.mlp.gate_proj',
    'model.layers.21.mlp.up_proj',
    'model.layers.21.mlp.down_proj',
    'model.layers.22.self_attn.q_proj',
    'model.layers.22.self_attn.k_proj',
    'model.layers.22.self_attn.v_proj',
    'model.layers.22.self_attn.o_proj',
    'model.layers.22.mlp.gate_proj',
    'model.layers.22.mlp.up_proj',
    'model.layers.22.mlp.down_proj',
    'model.layers.23.self_attn.q_proj',
    'model.layers.23.self_attn.k_proj',
    'model.layers.23.self_attn.v_proj',
    'model.layers.23.self_attn.o_proj',
    'model.layers.23.mlp.gate_proj',
    'model.layers.23.mlp.up_proj',
    'model.layers.23.mlp.down_proj',
    'model.layers.24.self_attn.q_proj',
    'model.layers.24.self_attn.k_proj',
    'model.layers.24.self_attn.v_proj',
    'model.layers.24.self_attn.o_proj',
    'model.layers.24.mlp.gate_proj',
    'model.layers.24.mlp.up_proj',
    'model.layers.24.mlp.down_proj',
    'model.layers.25.self_attn.q_proj',
    'model.layers.25.self_attn.k_proj',
    'model.layers.25.self_attn.v_proj',
    'model.layers.25.self_attn.o_proj',
    'model.layers.25.mlp.gate_proj',
    'model.layers.25.mlp.up_proj',
    'model.layers.25.mlp.down_proj',
    'model.layers.26.self_attn.q_proj',
    'model.layers.26.self_attn.k_proj',
    'model.layers.26.self_attn.v_proj',
    'model.layers.26.self_attn.o_proj',
    'model.layers.26.mlp.gate_proj',
    'model.layers.26.mlp.up_proj',
    'model.layers.26.mlp.down_proj',
    'model.layers.27.self_attn.q_proj',
    'model.layers.27.self_attn.k_proj',
    'model.layers.27.self_attn.v_proj',
    'model.layers.27.self_attn.o_proj',
    'model.layers.27.mlp.gate_proj',
    'model.layers.27.mlp.up_proj',
    'model.layers.27.mlp.down_proj',
    'lm_head']

def format_data(sample):
    image = sample["image"]
    # query = sample["query"]
    # Below flickr30k
    # query = "What is happening in the image"
    # answer = random.choice(sample["caption"])

    # VQA2
    query = sample["question"]
    answer = sample["multiple_choice_answer"]

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

# Constants
TOTAL_DATA_POINTS = 2000
TRAIN_TEST_SPLIT = 0.8
TRAIN_SIZE = int(TOTAL_DATA_POINTS * TRAIN_TEST_SPLIT)
TEST_SIZE = TOTAL_DATA_POINTS - TRAIN_SIZE
def create_dataset():
    """Load the VQA v2 dataset and split it into train and test subsets."""
    print("Loading VQA v2 dataset...")
    dataset = load_dataset("merve/vqav2-small", split="validation", cache_dir=setup_cache_dir())
    dataset = dataset.select(range(TOTAL_DATA_POINTS))  # Select first 2000 data points

    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(TRAIN_SIZE))
    test_dataset = dataset.select(range(TRAIN_SIZE, TOTAL_DATA_POINTS))

    return train_dataset, test_dataset

class DataCollator:
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

from benchmark.vqa2 import VQA_v2
def evaluate_model(model, dataset, benchmark_cls=VQA_v2):
    from benchmark.docvqa import DocVQA
    from benchmark.flickr30k import Flickr30k
    from datetime import datetime
    import pandas as pd

    benchmark = benchmark_cls(model, dataset)
    benchmark.evaluate()
    result = benchmark.results()

    print(f"Model: {model.get_model_name()}, Benchmark Metrics: {result}")
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

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def main(retained_var):
    print('#####', retained_var)
    dir_name = None
    if retained_var == 'prune':
        dir_name = 'pruned'
    elif retained_var == 'baseline':
        dir_name = 'baseline'
    else:
        dir_name = 'low_rank_' + str(retained_var)

    output_dir = os.path.join(os.getcwd(), f"models/qwen2_{dir_name}_vqa2")
    os.makedirs(output_dir, exist_ok=True)

    from model.qwen2 import Qwen2VL, CustomQwen2VL
    from low_rank import patch_model_using_metadata, save_metadata, get_metadata, replace_linear_with_low_rank

    qwen2 = Qwen2VL(quantization_mode=None)
    model, tokenizer, processor = qwen2.model, qwen2.tokenizer, qwen2.processor

    num_layers = len(model.model.layers)  # Total number of layers
    print(f"Original number of layers: {num_layers}")

    # Actual pruning/low-rank code would come here
    # ===
    print("Original model size:", get_model_size(model))
    if retained_var == 'prune':
        num_layers = len(model.model.layers)  # Total number of layers
        keep_layers = list(range(0, num_layers, 2))  # Keep every alternate layer
        print(f"Original number of layers: {num_layers}")
        model = prune_model_layers(model, keep_layers)
        print(f"Pruned number of layers: {len(model.model.layers)}")
    if retained_var == 'baseline':
        model = model
    else:
        model = replace_linear_with_low_rank(
            model, 
            retained_variance=retained_var,
            skip_patterns=get_skip_layers()
        )
    
    print("Compressed model size:", get_model_size(model))

    metadata_json = os.path.join(output_dir, "metadata.json")
    save_metadata(metadata_json)

    print("Saved metadata for low rank factorization")
    # ===

    # Prepare datasets for training
    print('Creating dataset splits')
    train_dataset, eval_dataset = create_dataset()

    train_dataset = [format_data(example) for example in train_dataset]
    print('Created dataset splits')

    # Recover from the latest checkpoint~
    def find_latest_checkpoint(output_dir):
        """Find the latest checkpoint in the output directory."""
        print(os.listdir(output_dir))
        checkpoints = [
            os.path.join(output_dir, d) for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        print('Candidate checkpoints', checkpoints)
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
        num_train_epochs=1,
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
        save_strategy="no",
    )
    os.makedirs(output_dir, exist_ok=True)

    data_collator = DataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    print("\nStarting fine-tuning...\n")
    start = time.time()
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    duration = time.time() - start
    print("\nFine-tuning completed. Time: ", duration)


    print("Saving model with device_map='auto' for offloading...")
    model.save_pretrained(output_dir, safe_serialization=False, max_shard_size="500MB", device_map="auto")
    print("Model saved successfully.")

    print("Saving processor")
    if not hasattr(processor, 'chat_template'):
        processor.chat_template = None

    print("Saving processor...")
    processor.save_pretrained(output_dir)
    print(f"Processor saved to {output_dir}")

    pytorch_model_path = os.path.join(output_dir, "pytorch_model.pt")
    torch.save(model.state_dict(), pytorch_model_path)

    custom_model = CustomQwen2VL(None, model, tokenizer, processor, f'{dir_name}_vqa2')
    # metadata = os.path.join(output_dir, 'metadata.json')
    # weights = os.path.join(output_dir, 'pytorch_model.pt')
    # custom_model = CustomQwen2VL.from_low_rank_path(metadata, weights)
    evaluate_model(custom_model, eval_dataset)

if __name__ == "__main__":
    configs = [0.8, 0.5, 'prune', 'baseline']
    # configs = ['baseline']
    for config in configs:
        main(config)