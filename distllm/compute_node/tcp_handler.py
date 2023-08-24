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
        self.slice_loader = SliceLoader()
        self.tensor_computer = TensorComputer()

        self.socket = socket
        self.context = RequestContext(self.registry, self.manager, self.name_gen,
                                      self.slice_loader, self.tensor_computer)

    def handle(self):
        msg, body = receive_message(self.socket)
        message = restore_message(msg, body)

        handler_cls = routes.get(message.get_message())
        handler = handler_cls(self.context)
        response = handler(message)
        response.send(self.socket)


class SliceLoader:
    def __call__(self, f):
        pass


class TensorComputer:
    def __call__(self, tensor):
        return tensor


class FailingSliceLoader(SliceLoader):
    def __call__(self, f):
        raise Exception('Something went wrong')


class FailingTensorComputer(TensorComputer):
    def __call__(self, tensor):
        raise Exception('Seomthing went wrong')


@dataclass
class RequestContext:
    registry: UploadRegistry
    manager: UploadManager
    name_gen: FunkyNameGenerator
    loader: SliceLoader
    tensor_computer: TensorComputer

    @classmethod
    def default(cls, uploads_dir='uploads', names=None):
        names = names or []
        registry = UploadRegistry(uploads_dir)
        manager = UploadManager(registry)
        name_gen = FunkyNameGenerator(names)
        loader = SliceLoader()
        tensor_computer = TensorComputer()
        return cls(registry, manager, name_gen, loader, tensor_computer)

    @classmethod
    def with_failing_loader(cls, uploads_dir='uploads', names=None):
        context = cls.default(uploads_dir, names)
        context.loader = FailingSliceLoader()
        return context
