from dataclasses import dataclass
from typing import Any
from distllm.compute_node.routes import routes
from distllm.compute_node.uploads import FunkyNameGenerator, UploadManager, UploadRegistry
from distllm.protocol import receive_message, restore_message


class TCPHandler:
    def __init__(self, socket, funky_names):
        self.registry = UploadRegistry('uploads')
        self.manager = UploadManager(self.registry)
        self.name_gen = FunkyNameGenerator(funky_names)
        self.load_slice = load_slice
        self.socket = socket
        self.context = RequestContext(self.registry, self.manager, self.name_gen)

    def handle(self):
        msg, body = receive_message(self.socket)
        message = restore_message(msg, body)

        handler_cls = routes.get(message.get_message())
        handler = handler_cls(self.context)
        response = handler(message)
        response.send(self.socket)


def load_slice(f):
    pass


@dataclass
class RequestContext:
    registry: UploadRegistry
    manager: UploadManager
    name_gen: FunkyNameGenerator
