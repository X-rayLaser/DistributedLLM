from datasets import load_dataset
from numpy import random

from .base import Command
from .common import get_llm


class PerplexityCommand(Command):
    name = 'perplexity'
    help = 'Calculate model perplexity on a given dataset'

    def configure_parser(self, parser):
        parser.add_argument('config', type=str, help='Path to the configuration file')

        parser.add_argument('--dataset', type=str, default='',
                    help='Dataset from Hugging Face dataset library')

        parser.add_argument('--dataset_name', type=str, default='',
                            help='Dataset name')

        parser.add_argument('--prompt', type=str, default='',
                            help='Text used to compute perplexity')

        parser.add_argument('--file', type=str, default='',
                            help='Path to a text file')

    def __call__(self, args):
        prompt = get_random_prompt(args)
        distributed_llm = get_llm(args.config)
        score = distributed_llm.perplexity(prompt)
        print(score)


def get_random_prompt(args):
    prompt = ''
    
    if args.dataset and args.dataset_name:
        ds = load_dataset(args.dataset, args.dataset_name, split="test")
        texts = ds["text"]
        large_texts = [text for text in texts if 1000 < len(text.strip()) < 5000]
        prompt = random.choice(large_texts)
        prompt = prompt.strip()[:500]
    elif args.prompt:
        prompt = args.prompt
    elif args.file:
        with open(args.file) as f:
            prompt = f.read()
    else:
        raise Exception('Expects either --prompt or --file arguments to be provided. Got none of them')

    return prompt
