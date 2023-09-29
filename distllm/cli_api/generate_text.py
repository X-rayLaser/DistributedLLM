import json
import time
from .base import Command
from ..control_center import Connection
from ..compute_node.slices import Tensor
from .common import get_llm
import numpy as np
import scipy


class GenerateTextCommand(Command):
    name = 'generate_text'
    help = 'Generate natural text using distributed LLM'

    def configure_parser(self, parser):
        parser.add_argument('config', type=str,
                            help='path to the configuration file')

        parser.add_argument('--prompt', type=str, default="",
                            help='Optional prompt')

        parser.add_argument('--num-tokens', type=int, default=100,
                            help='Maximum number of generated tokens')

        parser.add_argument('--temp', type=float, default=0.0,
                            help='Generation temperature')
        parser.add_argument('--rp', type=float, default=1.1,
                            help='Repetition penalty')
        
        parser.add_argument('--num-threads', type=int, default=2,
                            help='Number of threads used for computation')


    def __call__(self, args):
        distributed_llm = get_llm(args.config)

        start = time.time()
        after_first_token = time.time()
        n = 0
        for token_str in distributed_llm.generate(args.prompt, args.num_tokens,
                                                  temperature=args.temp, repeat_penalty=args.rp,
                                                  n_threads=args.num_threads):
            if n == 1:
                # to calculate tokens/sec, we must not include prompt evaluation time
                after_first_token = time.time()
            s = token_str
            n += 1
            print(f'{s}', end='', flush=True)

        now = time.time()
        elapsed = now - start
        speed = (n - 1) / (now - after_first_token)
        print()
        print(f"Generated {n} tokens in {elapsed:.2f} seconds. Average generation speed {speed:.2f} tokens/sec")