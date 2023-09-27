import os
import json
import socketserver
import socket
from .tcp_handler import TCPHandler, RequestContext
from distllm.compute_node import uploads
from distllm import protocol
from distllm.protocol import receive_message, restore_message


def run_server(host, port, uploads_dir, reverse_connect=False):
    registry_location = uploads.upload_registry.registry_data_path(uploads_dir)

    if os.path.isfile(registry_location):
        with open(registry_location) as f:
            registry_json = f.read()
        state_dict = json.loads(registry_json)
        uploads.upload_registry.load_state_dict(state_dict)
        
        print("Restored upload registry data from", registry_location)
    else:
        uploads.upload_registry.root = uploads_dir

    print("Initialized worker")

    if reverse_connect:
        connect_then_serve(host, port)
    else:
        with ThreadingTCPServer((host, port), MyTCPHandler) as server:
            server.serve_forever()

    print("Shutting down worker...")


def connect_then_serve(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        
        funky_names = ["orb", "pranker", "human", "alien", "sorcerer"]

        context = RequestContext.production(names=funky_names)
        my_handler = TCPHandler(sock, context)

        handshake(sock)

        serve_till_interruption(my_handler)


def handshake(sock):
    greeting_request = protocol.RequestGreeting()
    greeting_request.send(sock)

    msg, body = receive_message(sock)
    greeting_response = restore_message(msg, body)
    if greeting_response != protocol.ResponseGreeting():
        raise Exception("Failed to reverse connect: handshake failed")


def serve_till_interruption(handler):
    while True:
        try:
            handler.handle()  # blocking operation
        except KeyboardInterrupt:
            break


class MyTCPHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server) -> None:
        print("Creating instance of MyTCPHandler")
        funky_names = ["orb", "pranker", "human", "alien", "sorcerer"]

        context = RequestContext.production(uploads_dir='/distllm/uploads', names=funky_names)
        #context.registry.root = '/home/uploads'
        self.my_handler = TCPHandler(request, context)
        super().__init__(request, client_address, server)

    def handle(self):
        self.my_handler.handle()


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass
