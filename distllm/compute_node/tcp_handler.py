from dataclasses import dataclass
from typing import Any
from distllm.compute_node.routes import routes
from distllm.compute_node.slices import SliceContainer, NeuralComputationError, container
from distllm.compute_node.uploads import FunkyNameGenerator, UploadManager, UploadRegistry
from distllm.compute_node.uploads import upload_registry, upload_manager
from distllm.protocol import receive_message, restore_message
from distllm.utils import FakeFileSystemBackend, DefaultFileSystemBackend


class TCPHandler:
    def __init__(self, socket, context):
        self.socket = socket
        self.context = context

    def handle(self):
        while True:
            msg, body = receive_message(self.socket)
            message = restore_message(msg, body)
            print("Got message", msg)

            handler_cls = routes.get(message.get_message())
            handler = handler_cls(self.context)
            response = handler(message)

            print("About to send message", response.get_message())
            response.send(self.socket)

            if not hasattr(message, "keep_alive"):
                print("Quitting handle")
                break
            print("Keeping alive")


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
        fs_backend = FakeFileSystemBackend()
        manager.fs_backend = fs_backend
        slice_container = SliceContainer(fs_backend)
        return cls(registry, manager, name_gen, slice_container)

    @classmethod
    def with_failing_loader(cls, uploads_dir='uploads', names=None):
        context = cls.default(uploads_dir, names)
        fs_backend = FakeFileSystemBackend()
        context.slice_container = FailingSliceContainer(fs_backend)
        return context

    @classmethod
    def production(cls, uploads_dir='uploads', names=None):
        names = names or []
        registry = upload_registry
        manager = upload_manager
        name_gen = FunkyNameGenerator(names)
        slice_container = container
        manager.fs_backend = container.fs_backend
        return cls(registry, manager, name_gen, slice_container)
