import json
from generate_text import DistributedLLM


with open("./config.json") as f:
    s = f.read()

config = json.loads(s)

addresses = [tuple(addr) for addr in config["addresses"]]
distributed_llm = DistributedLLM(addresses,
                                 "../models/open_llama_3b_ggml-model-q4_0_extra_layers.bin")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute perplexity using distributed LLM')

    parser.add_argument('--prompt', type=str,
                        help='Text used to compute perplexity')

    parser.add_argument('--file', type=str,
                        help='Path to a text file whose content will be used to compute perplexity')
    
    args = parser.parse_args()
    prompt = ''
    if args.prompt:
        prompt = args.prompt
    elif args.file:
        with open(args.file) as f:
            prompt = f.read()
    else:
        raise Exception('Expects either --prompt or --file arguments to be provided. Got none of them')
    
    score = distributed_llm.perplexity(prompt)

    print(score)
