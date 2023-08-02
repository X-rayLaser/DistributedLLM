"""
Deploys compute node on a current device and starts up a TCP server.

Responsible for downloading an LLM, loading a consecutive subset of its layers into a memory and processing tensors.

Handles requests over TCP: reads a tensor from a socket, writes the resulting tensor back.
"""

import os
import struct
import json
from typing import Any


class LayerGroup:
    def __call__(self, inputs) -> Any:
        pass


import socketserver
import time
import threading
from queue import Queue
import uuid
import llm


class DummyModel:
    sleep_secs = 5

    def __call__(self, tensor):
        time.sleep(self.sleep_secs)
        return tensor


class Worker(threading.Thread):
    def __init__(self, model, max_jobs=10):
        super().__init__()
        self.lock = threading.Lock()
        self.queue = Queue(max_jobs)
        self.model = model
        self.results = {}

    def set_model(self, model):
        with self.lock:
            self.model = model

    def add_job(self, job):
        self.queue.put(job)

    def run(self) -> None:
        while True:
            job = self.queue.get()
            embeddings_in, job_id = job

            embeddings_out = self.model(embeddings_in)

            with self.lock:
                self.results[job_id] = embeddings_out

            self.queue.task_done()

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id)


class MyTCPHandler(socketserver.BaseRequestHandler):
    sleep_secs = 0.01

    def handle(self):
        command_line = self.receive_command()

        parts = command_line.split(' ')

        name = parts[0]

        print("Got ", command_line, parts, name)

        if name == "compute":
            print("receiving tensor")
            tensor = self.receive_tensor()
            job_id = uuid.uuid4().hex
            job = (tensor, job_id)
            worker.add_job(job)

            while worker.get_result(job_id) is None:
                time.sleep(self.sleep_secs)

            output_tensor = worker.get_result(job_id)
            print("output tensor len", len(output_tensor))

            tensor_bytes = bytearray();
            for t in output_tensor:
                tensor_bytes.extend(struct.pack('f', t))
            self.request.sendall(bytes(tensor_bytes))
        elif name == "setup_llm":
            self.receive_and_save_model(parts)

    def receive_command(self):
        s = self.request.recv(8)
        print("receive command", s)
        return s.strip().decode("utf-8")

    def receive_tensor(self):
        print("getting size bytes...")
        size_bytes = self.request.recv(4)
        print("got them")
        num_elements = struct.unpack('i', size_bytes)[0]
        print(num_elements)
        embeddings = []
        all_received_bytes = bytearray()
        while True:
            received = self.request.recv(1024)
            all_received_bytes.extend(received)
            if len(all_received_bytes) >= num_elements * 4:
                break
                
        for i in range(len(all_received_bytes) // 4):
            float_bytes = all_received_bytes[i * 4: (i + 1) * 4]
            emb = struct.unpack('f', float_bytes)[0]
            embeddings.append(emb)
            if len(embeddings) == num_elements:
                print("Received embeddings")
                return embeddings
                
        return embeddings

    def receive_and_setup_model(self, command):
        parts = command
        llm_name = parts[1]
        hparams = parts[2]
        layer_from = parts[3]
        layer_to = parts[4]

        llm_dir = llm_name

        # optionally, if llm directory exists and layer range is the same, return

        meta = {
            'layer_from': layer_from,
            'layer_to': layer_to,
            'llm_name': llm_name
        }

        os.makedirs(llm_dir, exist_ok=True)

        params_path = os.path.join(llm_dir, "hparams.json")
        with open(params_path, "w") as f:
            f.write(hparams)
        
        meta_path = os.path.join(llm_dir, "meta.json")
        with open(meta_path, "w") as f:
            f.write(json.dumps(meta))

        weights_path = os.path.join(llm_dir, "weights")        

        with open(weights_path, "wb") as f:
            while True:
                model_bytes = self.request.recv(1024)
                if model_bytes:
                    f.write(model_bytes)
                else:
                    break
        
        # construct model
        model = None

        # waiting for worker thread to stop before setting up a new model
        worker.queue.join()
        worker.set_model(model)
        worker.start()


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Deploy a compute node on this device')

    parser.add_argument('llm_slice', type=str,
                        help='Path to a slice of LLM')

    parser.add_argument('--host', type=str, default='localhost',
                        help='Host IP address')

    parser.add_argument('--port', type=int, default=9999,
                        help='Port number')

    args = parser.parse_args()

    llm.load_slice(args.llm_slice)
    
    print("Slice is loaded")

    class Slice:
        def __call__(self, tensor):
            return llm.propagate_forward(tensor)

    slice = Slice()
    worker = Worker(slice)
    worker.start()
    print("Initialized worker")

    with ThreadingTCPServer((args.host, args.port), MyTCPHandler) as server:
        server.serve_forever()

    print("Shutting down worker...")
    worker.queue.join()
