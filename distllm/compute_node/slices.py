from dataclasses import dataclass


@dataclass
class Tensor:
    shape: tuple
    values: list


class SliceContainer:
    def __init__(self):
        self.slice = None

    def load(self, f, metadata):
        if metadata.get('format') == 'test':
            data = f.read()
            k = data[0]
            b = data[1]
            self.slice = DummySlice(k, b)

    def forward(self, tensor: Tensor):
        if not self.is_loaded:
            raise SliceNotLoadedError()
        return self.slice(tensor)

    @property
    def info(self):
        return {}

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


container = SliceContainer()
