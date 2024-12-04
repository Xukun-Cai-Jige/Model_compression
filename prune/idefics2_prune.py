import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset
import random
from model.utils import setup_cache_dir

DEVICE = "cuda:0"


processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)


# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning

cache_dir = setup_cache_dir()
model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        #_attn_implementation="flash_attention_2", # Only available on A100 or H100
        cache_dir=cache_dir
    ).to(DEVICE)

#print(model)
#
#for name, module in model.named_modules():
#    if "attention" in name.lower():
#        print(name, module)


#heads_to_prune = {
#    0: [0, 1],  # Prune heads 0 and 1 in layer 0
#    1: [2, 3],  # Prune heads 2 and 3 in layer 1
#}
#
## Prune the attention heads
#for layer_idx, head_indices in heads_to_prune.items():
#    if layer_idx < len(model.model.vision_model.encoder.layers):
#        layer = model.model.vision_model.encoder.layers[layer_idx].self_attn
#
#        # Infer the number of attention heads
#        hidden_size = layer.q_proj.weight.shape[0]  # Hidden size
#        head_size = hidden_size // 8  # Adjust this value based on model inspection
#
#        # Zero out specified heads
#        for head_idx in head_indices:
#            start = head_idx * head_size
#            end = start + head_size
#            layer.q_proj.weight.data[:, start:end] = 0  # Zero out Q projection
#            layer.k_proj.weight.data[:, start:end] = 0  # Zero out K projection
#            layer.v_proj.weight.data[:, start:end] = 0  # Zero out V projection
#
#


import os

os.environ["TRANSFORMERS_CACHE"] = "/scratch/$USER/huggingface_cache"
os.environ["HF_HOME"] = "/scratch/$USER/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/$USER/huggingface_datasets"

train_dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", cache_dir="/scratch/$USER/huggingface_datasets")
train_dataset = train_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

eval_dataset = load_dataset("nielsr/docvqa_1200_examples", split="test", cache_dir="/scratch/$USER/huggingface_datasets")
eval_dataset = eval_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])


class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = example["query"]["en"]
            answer = random.choice(example["answers"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)

data_collator = MyDataCollator(processor)
sample_batch = data_collator([train_dataset[0]])
print(sample_batch.keys())  # Should include input_ids, attention_mask, pixel_values, labels
import os
output_dir = os.path.join(os.getcwd(), "model_pruned")
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True if torch.cuda.is_bf16_supported() else False,
    fp16=False,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    output_dir=output_dir,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset, # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory
)

trainer.train()

trainer.push_to_hub()

example = eval_dataset[5]
example
example["image"]
model.eval()

image = example["image"]
query = example["query"]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Answer briefly."},
            {"type": "image"},
            {"type": "text", "text": query["en"]}
        ]
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
print(generated_texts)
save_pruned = os.path.join(os.getcwd(), "model_pruned")
os.makedirs(save_pruned, exist_ok=True)
model.save_pretrained(save_pruned)
processor.save_pretrained(save_pruned)

print(f"Pruned model saved to: {save_pruned}")


