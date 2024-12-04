from datasets import load_dataset
from tqdm import tqdm
import os
from PIL import Image
from benchmark.utils import average_normalized_levenshtein_similarity, average_bert_score_f1_value


class ScienceQA_DEMO:
    def __init__(self, model):
        self.model = model
        self.processor = model.get_processor()
        self.model_type = self.model.get_model_name()
        self.cache_dir = os.path.join("datasets", "scienceqa")
        self.dataset = load_dataset("derek-thomas/ScienceQA", split="test", cache_dir=self.cache_dir)

        # Load only the first `num_examples` examples
        self.dataset = self.dataset.select(range(min(16, len(self.dataset))))

        self.answers_unique = []
        self.generated_texts_unique = []
        self.output_file = "output_scienceqa_demo.txt"

    def evaluate(self):
        EVAL_BATCH_SIZE = 1

        with open(self.output_file, "w") as f:
            for i in tqdm(range(0, len(self.dataset), EVAL_BATCH_SIZE)):
                examples = self.dataset[i: i + EVAL_BATCH_SIZE]

                # Ensure examples is iterable
                if not isinstance(examples, list):
                    examples = [examples]

                # Skip questions without images
                valid_examples = [
                    example for example in examples if example["image"] and example["image"][0] is not None
                ]

                if not valid_examples:
                    # Skip if no valid examples
                    continue
                #for example in valid_examples:
                #    print(example)
                # Process valid examples
                self.answers_unique.extend([
                    chr(ord('A') + int(example["answer"][0])) for example in valid_examples
                ])

                # Prepare inputs
                images = [
                    [example["image"][0]] for example in valid_examples
                ]
                questions = [{"en": self._format_question(example)} for example in valid_examples]

                for idx, example in enumerate(valid_examples):
                    # Show the image
                    img = images[idx][0]
                    img_path = f"science_image_{i + idx + 1}.png"
                    img.save(img_path)
                    print(f"Image saved as {img_path}")

                    # Save details to the file
                    f.write(f"Test Case {i + idx + 1}:\n")
                    f.write(f"Question: {self._format_question(example)}\n")
                    f.write(f"Expected Answer: {chr(ord('A') + int(example['answer'][0]))}\n")

                # Run model inference
                outputs = self.model.process_image_queries(images, questions)

                # Store generated texts
                self.generated_texts_unique.extend(outputs)

                for output in outputs:
                    f.write(f"Model Output: {output.strip().upper()}\n")
                    f.write("-" * 50 + "\n")

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

    def results(self):
        # Clean outputs for comparison
        self.generated_texts_unique = [g.strip().upper() for g in self.generated_texts_unique]  # Ensure consistency
        
        # Calculate accuracy
        correct = sum(
            1 for pred, truth in zip(self.generated_texts_unique, self.answers_unique) if pred == truth
        )
        total = len(self.answers_unique)
        accuracy = correct / total * 100

        # Save accuracy to the file
        with open(self.output_file, "a") as f:
            f.write(f"Accuracy: {accuracy:.2f}% ({correct}/{total})\n")

        # Calculate BERT score F1
        bert_score_f1 = average_bert_score_f1_value(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )

        with open(self.output_file, "a") as f:
            f.write(f"BERT Score F1: {bert_score_f1}\n")

        return accuracy, bert_score_f1
