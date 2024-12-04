import os
import torch
import torch.nn.utils.prune as prune
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
import random
import json
# Suggested DeepSpeed config (optional, not used here for simplicity)
DS_CONFIG_DICT = {
    'zero_optimization': {
        'stage': 2,
        'allgather_partitions': True,
        'allgather_bucket_size': 5e8,
        'overlap_comm': True,
        'reduce_scatter': True,
        'reduce_bucket_size': 5e8,
        'contiguous_gradients': True,
        'round_robin_gradients': True,
    },
    'bf16': {'enabled': True},
}

# Function to create the dataset
def create_dataset():
    train_dataset = load_dataset('nielsr/docvqa_1200_examples', split='train')
    train_dataset = train_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

    eval_dataset = load_dataset('nielsr/docvqa_1200_examples', split='test')
    eval_dataset = eval_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

    return train_dataset, eval_dataset

# Function to create the model
def create_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use BF16 precision
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',  # Use FlashAttention
    )
    return model

# Custom data collator for DocVQA
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
def setup_cache_dir():
    """Setup local cache directories for model and data"""
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set HuggingFace cache directories
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
    
    return cache_dir


def evaluate_model(model, processor, eval_dataset, output_dir):
    """Evaluate the model and save results"""
    print("Starting evaluation...")
    generated_texts = []
    answers = []
    
    for example in eval_dataset:
        image = example['image']
        question = example['query']['en']
        answers.append(example['answers'])

        # Create prompt
        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}\nAnswer briefly.',
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        # Prepare inputs
        inputs = processor(prompt, [image], return_tensors='pt').to("cuda")
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64
        )
        generated_text = processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        generated_texts.append(generated_text)

    # Save evaluation results
    evaluation_path = os.path.join(output_dir, "evaluation_results.json")
    with open(evaluation_path, 'w') as f:
        results = {"answers": answers, "generated_texts": generated_texts}
        json.dump(results, f, indent=4)
    
    print(f"Evaluation results saved to {evaluation_path}")



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
    model_name = "microsoft/Phi-3.5-vision-instruct"
    output_dir = os.path.join(os.getcwd(), "phi3_pruned")
    cache_dir = setup_cache_dir()

    # Load the processor and model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_dir,
    ).to("cuda")

    # Prune the model
    num_layers = len(model.model.layers)  # Total number of layers
    keep_layers = list(range(0, num_layers, 2))  # Keep every alternate layer
    print(f"Original number of layers: {num_layers}")
    model = prune_model_layers(model, keep_layers)
    print(f"Pruned number of layers: {len(model.model.layers)}")
    # Prepare datasets
    train_dataset = load_dataset('nielsr/docvqa_1200_examples', split='train', cache_dir=cache_dir)
    eval_dataset = load_dataset('nielsr/docvqa_1200_examples', split='test', cache_dir=cache_dir)

    # Define training arguments
    training_args = TrainingArguments(
        num_train_epochs=2,
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

    # Define a custom data collator (assumes `MiniDocVQADataCollator` is defined in your code)
    data_collator = MiniDocVQADataCollator(processor)

    # Initialize Trainer
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

    os.makedirs(output_dir, exist_ok=True)

# Save the model with safe_serialization=False to handle shared tensors
    #state_dict = model.state_dict()
    #state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
    print("Saving model with device_map='auto' for offloading...")
    model.save_pretrained(output_dir, safe_serialization=False, max_shard_size="500MB", device_map="auto")
    print("Model saved successfully.")
# Save the processor
    print("Saving processor")
    if not hasattr(processor, 'chat_template'):
        processor.chat_template = None

# Save the processor
    print("Saving processor...")
    processor.save_pretrained(output_dir)
    print(f"Processor saved to {output_dir}")

    evaluate_model(model, processor, eval_dataset, output_dir)
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()