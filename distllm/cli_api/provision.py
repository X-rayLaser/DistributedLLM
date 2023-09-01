import json
import os
import subprocess

from .base import Command

from distllm.control_center import ControlCenter, Connection


class ProvisionCommand(Command):
    name = 'provision'
    help = 'Distribute a given LLM across compute nodes'

    def configure_parser(self, parser):
        parser.add_argument('config_path', type=str,
                            help='Path to YAML configuration file')

    def __call__(self, args):
        
        with open(args.config_path) as f:
            s = f.read()

        config_dict = json.loads(s)
        
        nodes_map = config_dict["nodes_map"]
        model_id = config_dict["model_id"]
        location = config_dict["location"]
        metadata = config_dict["metadata"]
        clean_metadata(metadata)

        partition = [aslice for aslice in nodes_map.values()]
        convert_and_slice_model(model_id, location, partition, metadata)

        with open(os.path.join("models_registry", "registry.json")) as f:
            registry_dict = json.loads(f.read())

        slices = registry_dict[model_id]['slices']

        ivl_to_path = {}
        for d in slices:
            a = int(d['a'])
            b = int(d['b'])
            path = d['path']
            ivl_to_path[(a, b)] = path

        for address_str, (a, b) in nodes_map.items():
            ip, port = address_str.split(":")
            address = (ip, int(port))
            connection = Connection(address)

            aslice = (int(a), int(b))
            path = ivl_to_path[aslice]

            print(f"Pushing slice '{path}' to '{ip}:{port}'")
            slice_metadata = metadata.copy()
            slice_metadata['layer_from'] = a
            slice_metadata['layer_to'] = b

            file_size = os.path.getsize(path)
            with open(path, "rb") as f:
                d = connection.push_slice(f, model=model_id, metadata=slice_metadata,
                                          file_size=file_size, progress_bar=True)
            
            slice_name = d['file_name']
            print("Success")

            res = connection.load_slice(slice_name)
            print(f'Loaded slice {slice_name} in memory on {ip}:{port}', res)


def convert_and_slice_model(model_id, location, partition, metadata):
    registry_dir = "models_registry"
    os.makedirs(registry_dir, exist_ok=True)
    registry_file = os.path.join(registry_dir, "registry.json")

    tree = ModelsDirectoryTree(registry_dir, metadata)

    os.makedirs(tree.ggml_model_dir, exist_ok=True)

    if not os.path.exists(tree.ggml_model_file):
        convert_to_ggml(location, tree.ggml_model_file)

    if metadata["quantization"] and not os.path.exists(tree.target_model_file):
        os.makedirs(tree.target_model_dir, exist_ok=True)
        quantize(tree.ggml_model_file, tree.target_model_file, metadata["quantization"])

    os.makedirs(tree.partition_dir, exist_ok=True)

    extract_extra_layers(tree.target_model_file, tree.model_extra_layers)

    all_slices = []
    for model_slice in partition:
        a, b = model_slice
        slice_name = f'{a}_{b}.bin'

        slice_path = os.path.join(tree.partition_dir, slice_name)

        all_slices.append(dict(path=slice_path, a=a, b=b))
        if not os.path.exists(slice_path):
            make_slice(tree.target_model_file, a, b, slice_path)

    initialize_registry(registry_file)    
    update_registry(registry_file, model_id, metadata, tree.target_model_dir,
                    all_slices, tree.model_extra_layers)


def initialize_registry(registry_file):
    if not os.path.exists(registry_file):
        with open(registry_file, "w") as f:
            f.write(json.dumps({}))


def update_registry(registry_file, model_id, metadata, model_dir, slices, extra_layers_file):
    with open(registry_file) as f:
        s = f.read()

    registry_dict = json.loads(s)
    registry_dict[model_id] = {
        'metadata': metadata,
        'model_dir': model_dir,
        'slices': slices,
        'extra_layers_file': extra_layers_file
    }
    with open(registry_file, "w") as f:
        f.write(json.dumps(registry_dict))


def clean_metadata(metadata):
    model_name = metadata["name"]
    family = metadata["family"]
    size = metadata["size"]
    usage_class = metadata["usage_class"]
    metadata["usage_class"] = metadata["usage_class"] or usage_class
    usage_class = metadata["usage_class"]
    quantization = metadata["quantization"]

    validate_string(model_name)
    validate_family(family)
    validate_string(size)
    validate_string(usage_class)
    validate_quantization(quantization)


class ModelsDirectoryTree:
    def __init__(self, root, metadata) -> None:
        model_name = metadata["name"]
        family = metadata["family"]
        size = metadata["size"]
        usage_class = metadata["usage_class"]
        quantization = metadata["quantization"]

        base_dir = os.path.join(root, family, model_name, size, usage_class)
        ggml_model_dir = os.path.join(base_dir, "ggml_model")
        ggml_model_file = os.path.join(ggml_model_dir, "model.bin")

        if quantization:
            target_model_dir = os.path.join(base_dir, quantization)
        else:
            target_model_dir = ggml_model_dir

        target_model_file = os.path.join(target_model_dir, 'model.bin')

        self.ggml_model_file = ggml_model_file
        self.ggml_model_dir = ggml_model_dir
        self.target_model_dir = target_model_dir
        self.target_model_file = target_model_file
        self.partition_dir = os.path.join(target_model_dir, 'model_slices')

        self.model_extra_layers = os.path.join(self.partition_dir, "extra_layers.bin")


def validate_family(family):
    supported_ones = ["llama_v1", "llama_v2",]
    if family.lower() not in supported_ones:
        raise UnsupportedFamilyError(f'Got {family}, expected one of {supported_ones}')


def validate_quantization(quantization):
    if not quantization:
        return

    # todo: add remaining methods here
    supported_methods = ["q4_0", "q4_1"]
    if quantization not in supported_methods:
        raise UnsupportedQuantizationMethodError(
            f'Got {quantization}, expected one of {supported_methods}'
        )


def validate_string(s):
    import re
    if re.findall('[^a-zA-Z\d_]', s):
        raise InvalidStringError(s)


class UnsupportedQuantizationMethodError(Exception):
    pass


class UnsupportedFamilyError(Exception):
    pass


class InvalidStringError(Exception):
    pass


def convert_to_ggml(location, output_path):
    import sys
    sys.path.append("vendor/llama.cpp")
    import convert

    args = [f'--outfile={output_path}', location]
    convert.main(args)


def quantize(model, output_path, quantization):
    proc = subprocess.Popen(['vendor/llama.cpp/quantize', model, output_path, quantization],
                            stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print(out.decode('utf-8'))


def make_slice(model, a, b, output_path):
    a = str(a)
    b = str(b)
    proc = subprocess.Popen(['distllm/slice_model', 'slice', model, a, b, output_path],
                            stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print(out.decode('utf-8'))


def extract_extra_layers(model, output_path):
    proc = subprocess.Popen(['distllm/slice_model', 'extra_layers', model, output_path],
                            stdout=subprocess.PIPE)
    out, err = proc.communicate()
    print(out.decode('utf-8'))
