from datasets import load_dataset

# Load the dataset
dataset_name = 'nielsr/docvqa_1200_examples'
dataset = load_dataset(dataset_name)

# Print dataset structure
print("Dataset structure:", dataset)

# Inspect the train split (first few examples)
print("\nFirst few examples from the train split:")
print(dataset['train'][:5])  # Adjust the number as needed

# Inspect a single example in detail
print("\nSingle example:")
print(dataset['train'][0])

# List all available columns
print("\nColumns in the dataset:")
print(dataset['train'].column_names)