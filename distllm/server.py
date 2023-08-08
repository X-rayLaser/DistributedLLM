import json
from flask import Flask, request
from generate_text import DistributedLLM



with open("./config.json") as f:
    s = f.read()

config = json.loads(s)

addresses = [tuple(addr) for addr in config["addresses"]]
distributed_llm = DistributedLLM(addresses,
                                 "../models/open_llama_3b_ggml-model-q4_0_extra_layers.bin")

app = Flask(__name__)


@app.route('/generate')
def generate():
    prompt = request.args.get('prompt')
    max_tokens = int(request.args.get('max-tokens'))

    gen = distributed_llm.generate(prompt, max_tokens)
    s = ''.join(gen)
    return s


@app.route('/perplexity')
def perplexity():
    prompt = request.args.get('prompt')
    score = distributed_llm.perplexity(prompt)
    return str(score)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
