from dataclasses import dataclass
from distllm.utils import DefaultFileSystemBackend

@dataclass
class Tensor:
    shape: tuple
    values: list


class SliceContainer:
    def __init__(self, fs_backend):
        self.fs_backend = fs_backend
        self.slice = None
        self.metadata = None

    def load(self, slice_path, metadata):
        self.metadata = metadata

        if metadata.get('format') == 'test':
            f = self.fs_backend.open_file(slice_path, mode='rb')
            data = f.read()
            f.close()
            k = data[0]
            b = data[1]
            self.slice = DummySlice(k, b)
            return
        

    def forward(self, tensor: Tensor):
        if not self.is_loaded:
            raise SliceNotLoadedError()
        return self.slice(tensor)

    @property
    def info(self):
        return self.metadata

    @property
    def is_loaded(self):
        return self.slice is not None


class SliceNotLoadedError(Exception):
    pass


class NeuralComputationError(Exception):
    pass


class ModelSlice:
    def __call__(self, tensor: Tensor):
        pass


class DummySlice(ModelSlice):
    def __init__(self, k, b):
        self.k = k
        self.b = b

    def __call__(self, tensor: Tensor):
        new_values = [self.k * v + self.b for v in tensor.values]
        return Tensor(tensor.shape, new_values)


class GGMLSlice(ModelSlice):
    def __init__(self, file_path):
        self.path  = file_path

        import llm

        llm.load(self.path)

    def __call__(self, tensor: Tensor):
        return super().__call__(tensor)


fs_backend = DefaultFileSystemBackend()
container = SliceContainer(fs_backend)
