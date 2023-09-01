import os
import json
import socketserver
from socketserver import BaseServer
from .tcp_handler import TCPHandler, RequestContext
from distllm.utils import DefaultFileSystemBackend
from distllm.compute_node import uploads

def run_server(host, port, uploads_dir):
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
    with ThreadingTCPServer((host, port), MyTCPHandler) as server:
        server.serve_forever()

    print("Shutting down worker...")


class MyTCPHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server) -> None:
        print("Creating instance of MyTCPHandler")
        funky_names = ["orb", "pranker", "human", "alien", "sorcerer"]

        context = RequestContext.production(uploads_dir='/distllm/uploads', names=funky_names)
        #context.registry.root = '/home/uploads'
        self.my_handler = TCPHandler(request, context)
        super().__init__(request, client_address, server)

    def handle(self):
        funky_names = ["orb", "pranker", "human", "alien", "sorcerer"]
        #self.my_handler = TCPHandler(self.request, funky_names=funky_names)
        #self.my_handler.manager.fs_backend = DefaultFileSystemBackend()
        self.my_handler.handle()


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass
