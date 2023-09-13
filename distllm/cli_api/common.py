import json
from ..control_center import Connection
import llm

import numpy as np
import scipy


def get_llm(config_path):
    with open(config_path) as f:
        s = f.read()
    config = json.loads(s)

    items = list(config['nodes_map'].items())

    model_id = config["model_id"]
    load_all_slices(model_id, items)

    sorted_items = sorted(items, key=lambda t: t[1])
    sorted_nodes = []
    for address_str, _ in sorted_items:
        sorted_nodes.append(parse_address(address_str))

    with open("models_registry/registry.json") as f:
        s = f.read()
    models_registry = json.loads(s)
    return DistributedLLM(sorted_nodes, models_registry[model_id]['extra_layers_file'])


def load_all_slices(model_id, nodes_with_slices):
    for address_str, (a, b) in nodes_with_slices:
        load_one_slice(model_id, address_str, a, b)


def load_one_slice(model_id, address_str, a, b):
    connection = Connection(address=parse_address(address_str))

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

def parse_address(address):
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

    def perplexity(self, text):
        self.clear_context()
        extra_layers_path = self.extra_layers_path
        tokens = llm.tokenize_prompt(extra_layers_path, text)

        embeddings = llm.prepare_embeddings(extra_layers_path, tokens[:-1])

        embeddings = self.propagate_tensor(embeddings)

        tokens_shifted = tokens[1:]
        all_logits = True
        logits = llm.get_logits(extra_layers_path, embeddings, all_logits)
        num_tokens_out = len(tokens) - 1
        assert len(logits) % num_tokens_out == 0
        logits = np.array(logits).reshape(num_tokens_out, -1)

        pmf = scipy.special.softmax(logits, axis=1)

        rows = np.arange(num_tokens_out)
        cols = tokens_shifted

        probabilities = pmf[rows, cols]

        nll = 0

        for t in range(num_tokens_out):
            nll -= np.log(probabilities[t])
        
        return np.exp(nll / num_tokens_out)

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
