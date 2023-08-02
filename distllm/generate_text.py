"""
Uses prepared Distributed LLM Compute Service to generate text from LLM.
"""


import socket
import struct
import llm


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

    tokens = llm.tokenize_prompt(extra_layers_path, prompt)
   
    num_steps = args.max_tokens

    output = ""
    for t in range(num_steps):
        embeddings = llm.prepare_embeddings(extra_layers_path, tokens)
        embeddings = propagate_tensor(address1, embeddings)
        embeddings = propagate_tensor(address2, embeddings)
        token = llm.get_next_token(extra_layers_path, embeddings)
        token_str = llm.decode_token(extra_layers_path, token)
        tokens.clear()
        tokens.append(token)

        s = token_str
        output += s.strip('\n')
        print(f'{s}', end='', flush=True)

    print()
