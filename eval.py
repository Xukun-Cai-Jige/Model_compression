from model.qwen2 import Qwen2VL
from model.idefics2 import Idefics2
from model.phi3 import Phi3_5
from model.llava import VideoLLava

# setup cli args
import argparse
import pandas as pd

from datetime import datetime
import os

def main(args):
    if os.path.exists("results.csv"):
        df = pd.read_csv("results.csv")
    else:
        df = pd.DataFrame(columns=["timestamp", "model_name", "benchmark", "accuracy", "memory_utilization", "model_runtime", "additional_results"])

    # Load the model based on the input argument
    if args.model_name == "Qwen/Qwen2-VL-2B-Instruct":
        model = Qwen2VL(quantization_mode=args.quantization_mode)
    elif args.model_name == "HuggingFaceM4/idefics2-8b":
        model = Idefics2(quantization_mode=args.quantization_mode)
    elif args.model_name == "microsoft/Phi-3.5-vision-instruct":
        model = Phi3_5(quantization_mode=args.quantization_mode)
    elif args.model_name == "LanguageBind/Video-LLaVA-7B-hf":
        model = VideoLLava(quantization_mode=args.quantization_mode)
    elif args.model_name == "MAGAer13/mplug-owl-llama-7b-video":
        from model.mplug import Mplug
        model = Mplug(quantization_mode=args.quantization_mode)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    # Load the benchmark based on the input argument
    if args.benchmark_name == "docvqa":
        from benchmark.docvqa import DocVQA
        benchmark = DocVQA(model)
    elif args.benchmark_name == "vqa2":
        from benchmark.vqa2 import VQA_v2
        benchmark = VQA_v2(model)
    elif args.benchmark_name == "scienceqa":
        from benchmark.scienceqa import ScienceQA
        benchmark = ScienceQA(model)
    elif args.benchmark_name == "docvqa_demo":
        from benchmark.docvqa_demo import DocVQA_DEMO
        benchmark = DocVQA_DEMO(model)
    elif args.benchmark_name == "vqa2_demo":
        from benchmark.vqa2_demo import VQA_v2_DEMO
        benchmark = VQA_v2_DEMO(model)
    elif args.benchmark_name == "scienceqa_demo":
        from benchmark.scienceqa_demo import ScienceQA_DEMO
        benchmark = ScienceQA_DEMO(model)
    elif args.benchmark_name == 'flickr30k':
        from benchmark.flickr30k import Flickr30k
        benchmark = Flickr30k(model)
    else:
        raise ValueError(f"Benchmark {args.benchmark_name} not supported")

    # Run the evaluation and display results
    benchmark.evaluate()
    result = benchmark.results()

    print(f"Model: {model.get_model_name()}, Benchmark: {args.benchmark_name}, Accuracy: {result}")
    now = datetime.now()

    formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Save the results to a CSV file
    df = pd.concat([df, pd.DataFrame([[formatted_timestamp, model.get_model_name(), args.benchmark_name, result, model.get_model_size(), model.get_average_processing_time() , ""]], columns=df.columns)], ignore_index=True)
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization_mode", type=int, default=16, help="Quantization mode for the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--benchmark_name", type=str, default="docvqa", help="Benchmark name to run (e.g., scienceqa, vqa2).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name to evaluate.")

    args = parser.parse_args()

    main(args)
