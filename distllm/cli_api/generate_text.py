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

        self.load_all_slices(config["model_id"], items)

        sorted_items = sorted(items, key=lambda t: t[1])
        sorted_nodes = []
        for address_str, _ in sorted_items:
            sorted_nodes.append(self.parse_address(address_str))

        with open("models_registry/registry.json") as f:
            s = f.read()
        models_registry = json.loads(s)
        distributed_llm = DistributedLLM(sorted_nodes, models_registry['my_open_llama']['extra_layers_file'])

        for token_str in distributed_llm.generate(args.prompt, args.num_tokens,
                                                  temperature=args.temp, repeat_penalty=args.rp):
            s = token_str
            print(f'{s}', end='', flush=True)

        print()

    def load_all_slices(self, model_id, nodes_with_slices):
        for address_str, (a, b) in nodes_with_slices:
            self.load_one_slice(model_id, address_str, a, b)

    def load_one_slice(self, model_id, address_str, a, b):
        connection = Connection(address=self.parse_address(address_str))

        status = connection.get_status()

        if status['status'] == 'up':
            meta = status['metadata']
            if model_id == meta['model'] and a == meta['layer_from'] and b == meta['layer_to']:
                print(f"Slice {model_id}:({a}, {b}) is already loaded")
                return
            # todo: handle case where wrong slice is currently loaded

        named_slices = connection.list_all_slices()

        for s in named_slices:
            if model_id == s['model'] and a == s['layer_from'] and b == s['layer_to']:
                slice_name = s['name']
                connection.load_slice(slice_name)
                print(f"Loaded slice {model_id}:({a}, {b}) into memory on node {address_str}")
                return
        
        print(f"Could not find a slice to load on node at {address_str}")

    def parse_address(self, address):
        host, port = address.split(":")
        port = int(port)
        return host, port


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
        self.clear_context()
        extra_layers_path = self.extra_layers_path
        tokens = llm.tokenize_prompt(extra_layers_path, prompt)

        sampler = Sampler(temperature, repeat_penalty)
        all_logits = False
        for _ in range(max_steps):
            embeddings = llm.prepare_embeddings(extra_layers_path, tokens)
            embeddings = self.propagate_tensor(embeddings)
            logits = llm.get_logits(extra_layers_path, embeddings, all_logits)

            token_id = sampler(logits)

            token_str = llm.decode_token(extra_layers_path, token_id)
            tokens.clear()
            tokens.append(token_id)
            yield token_str

    def clear_context(self):
        for host_with_port in self.addresses:
            connection = Connection(host_with_port)
            connection.clear_context()

    def propagate_tensor(self, embeddings):
        shape = (1, len(embeddings))
        for host_with_port in self.addresses:
            connection = Connection(host_with_port)
            res = connection.propagate_forward(embeddings, shape)
            embeddings = res['values']
        return embeddings

