from dataclasses import dataclass
from typing import Any
from distllm.compute_node.routes import routes
from distllm.compute_node.slices import SliceContainer, NeuralComputationError, container
from distllm.compute_node.uploads import FunkyNameGenerator, UploadManager, UploadRegistry
from distllm.compute_node.uploads import upload_registry, upload_manager
from distllm.protocol import receive_message, restore_message


class TCPHandler:
    def __init__(self, socket, funky_names):
        self.registry = upload_registry
        self.manager = upload_manager
        self.name_gen = FunkyNameGenerator(funky_names)
        self.slice_container = container
        self.socket = socket
        self.context = RequestContext(upload_registry, upload_manager, self.name_gen,
                                      self.slice_container)

    def handle(self):
        msg, body = receive_message(self.socket)
        message = restore_message(msg, body)
        print("Got message", msg, body)

        handler_cls = routes.get(message.get_message())
        handler = handler_cls(self.context)
        response = handler(message)

        print("About to send message", response.get_message(), response.get_body())
        response.send(self.socket)


class SliceLoader:
    def __call__(self, f):
        pass


class TensorComputer:
    def __call__(self, tensor):
        return tensor


class FailingSliceContainer(SliceContainer):
    def load(self, f, metadata):
        raise Exception('Something went wrong')

    def forward(self, tensor):
        raise NeuralComputationError('Something went wrong')


@dataclass
class RequestContext:
    registry: UploadRegistry
    manager: UploadManager
    name_gen: FunkyNameGenerator
    slice_container: SliceContainer

    @classmethod
    def default(cls, uploads_dir='uploads', names=None):
        names = names or []
        registry = UploadRegistry(uploads_dir)
        manager = UploadManager(registry)
        name_gen = FunkyNameGenerator(names)
        slice_container = SliceContainer()
        return cls(registry, manager, name_gen, slice_container)

    @classmethod
    def with_failing_loader(cls, uploads_dir='uploads', names=None):
        context = cls.default(uploads_dir, names)
        context.slice_container = FailingSliceContainer()
        return context
