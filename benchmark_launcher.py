import argparse

from eval import main

if __name__ == "__main__":
    quantization_modes = [4,8, 32]
    # , "docvqa"
    models = ["Qwen/Qwen2-VL-2B-Instruct", "LanguageBind/Video-LLaVA-7B-hf", "HuggingFaceM4/idefics2-8b", "microsoft/Phi-3.5-vision-instruct"]
    benchmarks = ["vqa2", "scienceqa", "docvqa"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization_mode", type=int, default=16, help="Quantization mode for the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--benchmark_name", type=str, default="docvqa", help="Benchmark name to run (e.g., scienceqa, vqa2).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name to evaluate.")

    args = parser.parse_args()

    for model in models:
        for quantization_mode in quantization_modes:
            for benchmark in benchmarks:
                print("Running benchmark {} for model {} with quantization mode {}".format(benchmark, model, quantization_mode))
                args.model_name = model
                args.quantization_mode = quantization_mode
                args.benchmark_name = benchmark
                main(args)