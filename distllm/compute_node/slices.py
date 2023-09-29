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

        self.slice = GGMLSlice(slice_path)

    def forward(self, tensor: Tensor, n_threads: int):
        if not self.is_loaded:
            raise SliceNotLoadedError()
        return self.slice(tensor, n_threads)

    def clear_context(self):
        if self.is_loaded:
            self.slice.clear_context()

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

    def clear_context(self):
        pass


class DummySlice(ModelSlice):
    def __init__(self, k, b):
        self.k = k
        self.b = b

    def __call__(self, tensor: Tensor, n_threads: int):
        new_values = [self.k * v + self.b for v in tensor.values]
        return Tensor(tensor.shape, new_values)


class GGMLSlice(ModelSlice):
    def __init__(self, file_path):
        import llm
        self.path  = file_path

        llm.load_slice(self.path)

    def __call__(self, tensor: Tensor, n_threads: int):
        import llm
        new_values = llm.propagate_forward(tensor.values, n_threads)
        return Tensor(tensor.shape, new_values)

    def clear_context(self):
        import llm
        res = llm.clear_context()
        if res != 0:
            raise Exception("Error occurred when clearing context")


fs_backend = DefaultFileSystemBackend()
container = SliceContainer(fs_backend)
