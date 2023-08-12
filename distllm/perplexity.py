import json
from numpy import random
from generate_text import DistributedLLM
from datasets import load_dataset

with open("./config.json") as f:
    s = f.read()

config = json.loads(s)

addresses = [tuple(addr) for addr in config["addresses"]]
distributed_llm = DistributedLLM(addresses,
                                 "../models/open_llama_3b_ggml-model-q4_0_extra_layers.bin")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute perplexity using distributed LLM')

    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset from Hugging Face dataset library')

    parser.add_argument('--dataset_name', type=str, default='',
                        help='Dataset name')


    parser.add_argument('--prompt', type=str, default='',
                        help='Text used to compute perplexity')

    parser.add_argument('--file', type=str, default='',
                        help='Path to a text file')
    
    args = parser.parse_args()
    prompt = ''
    
    if args.dataset and args.dataset_name:
        ds = load_dataset(args.dataset, args.dataset_name, split="test")
        texts = ds["text"]
        large_texts = [text for text in texts if 1000 < len(text.strip()) < 5000]
        prompt = random.choice(large_texts)
        prompt = prompt.strip()[:750]
    elif args.prompt:
        prompt = args.prompt
    elif args.file:
        with open(args.file) as f:
            prompt = f.read()
    else:
        raise Exception('Expects either --prompt or --file arguments to be provided. Got none of them')
    
    print("Evaluating perplexity using prompt:\n")
    print(prompt)
    score = distributed_llm.perplexity(prompt)

    print(score)
