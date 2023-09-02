import json
from .base import Command
from ..control_center import Connection
from ..compute_node.slices import Tensor
import llm

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

    def __call__(self, args):
        with open(args.config) as f:
            s = f.read()
        config = json.loads(s)

        items = list(config['nodes_map'].items())

        sorted_items = sorted(items, key=lambda t: t[1])
        sorted_nodes = []
        for address_str, _ in sorted_items:
            host, port = address_str.split(":")
            port = int(port)
            sorted_nodes.append((host, port))

        with open("models_registry/registry.json") as f:
            s = f.read()
        models_registry = json.loads(s)
        distributed_llm = DistributedLLM(sorted_nodes, models_registry['my_open_llama']['extra_layers_file'])

        for token_str in distributed_llm.generate(args.prompt, args.num_tokens,
                                                  temperature=args.temp, repeat_penalty=args.rp):
            s = token_str
            print(f'{s}', end='', flush=True)

        print()


class Sampler:
    def __init__(self, temperature=0.7, repeat_penalty=1.1):
        self.T = temperature
        self.penalty = repeat_penalty
        self.previous_ids = []
        self.eps = 10**(-5)

    def __call__(self, logits):
        logits = np.array(logits)
        
        size = len(logits)

        token_ids = np.arange(size)

        mask = np.isin(token_ids, self.previous_ids)
        penalties = (mask * self.penalty + ~mask) * (self.T + self.eps)
        logits = logits / penalties

        probs = scipy.special.softmax(logits)
        token_id = np.random.choice(token_ids, p=probs)
        token_id = int(token_id)
        self.previous_ids.append(token_id)
        return token_id


class DistributedLLM:
    def __init__(self, addresses, extra_layers_path):
        self.addresses = addresses
        self.extra_layers_path = extra_layers_path

    def generate(self, prompt, max_steps=200, temperature=0.0, repeat_penalty=1.1):
        extra_layers_path = self.extra_layers_path
        tokens = llm.tokenize_prompt(extra_layers_path, prompt)

        sampler = Sampler(temperature, repeat_penalty)

        for _ in range(max_steps):
            embeddings = llm.prepare_embeddings(extra_layers_path, tokens)
            embeddings = self.propagate_tensor(embeddings)
            logits = llm.get_logits(extra_layers_path, embeddings)

            token_id = sampler(logits)

            token_str = llm.decode_token(extra_layers_path, token_id)
            tokens.clear()
            tokens.append(token_id)
            yield token_str

    def propagate_tensor(self, embeddings):
        shape = (1, len(embeddings))
        for host_with_port in self.addresses:
            connection = Connection(host_with_port)
            res = connection.propagate_forward(embeddings, shape)
            embeddings = res['values']
        return embeddings

