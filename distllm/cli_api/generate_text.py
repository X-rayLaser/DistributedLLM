import json
from .base import Command
from ..control_center import Connection
from ..compute_node.slices import Tensor
import llm


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

    def __call__(self, args):
        with open(args.config) as f:
            s = f.read()
        config = json.loads(s)

        items = list(config['nodes_map'].items())
        print(items)
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

        for token_str in distributed_llm.generate(args.prompt, args.num_tokens):
            s = token_str
            print(f'{s}', end='', flush=True)

        print()


class DistributedLLM:
    def __init__(self, addresses, extra_layers_path):
        self.addresses = addresses
        self.extra_layers_path = extra_layers_path

    def generate(self, prompt, max_steps=200):
        extra_layers_path = self.extra_layers_path
        tokens = llm.tokenize_prompt(extra_layers_path, prompt)

        for _ in range(max_steps):
            embeddings = llm.prepare_embeddings(extra_layers_path, tokens)
            embeddings = self.propagate_tensor(embeddings)
            token = llm.get_next_token(extra_layers_path, embeddings)
            token_str = llm.decode_token(extra_layers_path, token)
            tokens.clear()
            tokens.append(token)
            yield token_str

    def propagate_tensor(self, embeddings):
        shape = (1, len(embeddings))
        for host_with_port in self.addresses:
            connection = Connection(host_with_port)
            res = connection.propagate_forward(embeddings, shape)
            embeddings = res['values']
        return embeddings

