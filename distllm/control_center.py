import json
from dataclasses import dataclass
from distllm import utils, protocol


class ControlCenter:
    def __init__(self, nodes_map):
        """Takes a dict that maps verbose name to tuples (ip_address, port) 
        specifying location of each compute node"""
        self.nodes_map = nodes_map

        nodes = {}
        for name, (ip_address, port) in self.nodes_map.items():
            nodes[name] = {
                'connectivity': True,
                'ip': ip_address,
                'port': port,
                'status': 'brand_new',
                'errors': [],
                'slice': None
            }
        
        self.status = {
            'ready': False,
            'model': None,
            'nodes': nodes
        }

    def push_model(self, model_name, slices, meta_data=None):
        """Push slices to their destination nodes"""

        candidates = set(slices.keys())
        node_names = set(self.nodes_map.keys())
        if candidates != node_names:
            raise NodeProvisioningError

        nodes = {}
        for name in self.status['nodes'].copy():
            nodes[name] = self.status['nodes'][name].copy()
            nodes[name]['status'] = 'up'
            slice = slices[name]
            nodes[name]['slice'] = [slice.layer_from, slice.layer_to]

        self.status = {
            'ready': True,
            'model': {
                'pseudoname': model_name,
                'family': None,
                'class': None,
                'size': None
            },
            'nodes': nodes
        }

    def list_models(self):
        """Return information about each model pushed to and distributed on compute network"""

    def set_model(self, name):
        """Set a given model for computation"""

    def get_status(self):
        """Ping every node and query it's readiness status"""
        return self.status

    def propagate_forward(self, embeddings_tensor):
        """Pass embeddings tensor through all slices and return resulting tensor"""

    def get_topology(self):
        """Topology of compute network: maps compute nodes to their slices"""


@dataclass
class ModelSlice:
    blob: bytes
    layer_from: int
    layer_to: int


class NodeProvisioningError(Exception):
    pass


class Connection:
    def __init__(self, address):
        self.address = address
        self.connect = connect

    def push_slice(self, slice, name, layer_range):
        """
        Send a model slice to a remote compute node.
        """

    def list_all_slices(self):
        """List all slices pushed to the compute node"""
        socket = self.connect(self.address)
        message_out = protocol.RequestAllSlices()
        message_out.send(socket)

        message_text, body = protocol.receive_message(socket)
        message = protocol.restore_message(message_text, body)
        return json.loads(message.slices_json)

    def load_slice(self, name):
        """Load to memory a model slice with a given name"""
        socket = self.connect(self.address)
        message_out = protocol.RequestLoadSlice(name=name)
        message_out.send(socket)

        message_text, body = protocol.receive_message(socket)
        message = protocol.restore_message(message_text, body)
        if message_text == 'operation_failure':
            raise OperationFailedError('')
        return message.get_body()

    def get_status(self):
        """Receive compute node readiness status and meta information about the slice"""
        socket = self.connect(self.address)
        message_out = protocol.RequestStatus()
        message_out.send(socket)
        message_text, body = protocol.receive_message(socket)
        message = protocol.restore_message(message_text, body)
        return json.loads(message.status_json)

    def propagate_forward(self, tensor):
        """Send a tensor to a remote node and propagate it forward through layers of the slice"""


class OperationFailedError(Exception):
    pass


def connect(address):
    """Connects to the remote compute node, returns a socket"""
