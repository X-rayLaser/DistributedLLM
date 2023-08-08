"""
Uses prepared Distributed LLM Compute Service to generate text from LLM.
"""


import socket
import struct
import llm
import numpy as np
import scipy

def propagate_tensor(address, embeddings):
    barray = bytearray()
    for e in embeddings:
        barray.extend(struct.pack('f', e))
    
    num_elements = len(embeddings)
    embeddings_size = struct.pack('i', num_elements)
    
    data = bytes("compute\n", "utf-8") + embeddings_size + bytes(barray)

    size_of_float = 4
    chunk_size = 1024

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)
        sock.sendall(data)

        all_received = bytes()
        result = []
        while True:
            received = sock.recv(chunk_size)
            if len(all_received) >= len(embeddings) * size_of_float:
                break
            all_received += received

            num_elements = len(received) // size_of_float
            for i in range(num_elements):
                value_bytes = received[i * size_of_float:(i + 1) * size_of_float]
                if value_bytes == b'':
                    raise Exception('Unexpected empty byte string')
                x = struct.unpack('f', value_bytes)[0]
                result.append(x)
    return result


def softmax(v):
    return scipy.special.softmax(v, axis=1)


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

    def perplexity(self, text):
        extra_layers_path = self.extra_layers_path
        tokens = llm.tokenize_prompt(extra_layers_path, text)

        embeddings = llm.prepare_embeddings(extra_layers_path, tokens[:-1])

        embeddings = self.propagate_tensor(embeddings)

        tokens_shifted = tokens[1:]
        logits = llm.get_logits(extra_layers_path, embeddings)
        num_tokens_out = len(tokens) - 1
        assert len(logits) % num_tokens_out == 0
        logits = np.array(logits).reshape(num_tokens_out, -1)

        pmf = softmax(logits)

        rows = np.arange(num_tokens_out)
        cols = tokens_shifted

        probabilities = pmf[rows, cols]

        nll = 0

        for t in range(num_tokens_out):
            nll -= np.log(probabilities[t])
        
        return np.exp(nll / num_tokens_out)

    def propagate_tensor(self, embeddings):
        for host_with_port in self.addresses:
            embeddings = propagate_tensor(host_with_port, embeddings)
        return embeddings


def parse_address(address):
    host, port = address.split(':')
    return host, int(port)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate text using distributed LLM')

    parser.add_argument('address1', type=str,
                        help='Colon separated IP address and port (IP:PORT) of first compute node')
    parser.add_argument('address2', type=str,
                        help='Colon separated IP address and port (IP:PORT) of second compute node')
    parser.add_argument('extra_layers', type=str, help='Path to extra layers')
    parser.add_argument('prompt', type=str, help='Text prompt')
    parser.add_argument('--max-tokens', type=int, default=200,
                        help='Maximum number of new tokens to generate')

    args = parser.parse_args()
    address1 = parse_address(args.address1)
    address2 = parse_address(args.address2)
    prompt = args.prompt
    extra_layers_path = args.extra_layers

    num_steps = args.max_tokens

    distributed_llm = DistributedLLM([address1, address2], extra_layers_path)

    for token_str in distributed_llm.generate(prompt, num_steps):
        s = token_str
        print(f'{s}', end='', flush=True)

    print()
