import json
from dataclasses import dataclass
import hashlib
from tqdm import tqdm
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


# todo: connect only once (in init)
# todo: close method
# todo: context manager
class Connection:
    def __init__(self, address):
        self.address = address
        self.connect = connect
        self.disconnect = disconnect

    def push_slice(self, f, model, metadata=None, chunk_size=1024*1024, file_size=None, progress_bar=False):
        """
        Send a model slice to a remote compute node.
        """

        # todo: metadata must contain fields: layer_from, layer_to

        metadata = metadata or {}

        all_metadata = {
            'type': 'slice',
            'model': model
        }

        all_metadata.update(metadata)

        return self.push_file(f, all_metadata, chunk_size, file_size, progress_bar)

    def push_file(self, f, metadata=None, chunk_size=1024*1024, file_size=None, progress_bar=False):
        """
        Send a file to a remote compute node
        """

        # todo: ensure renewing socket before sending each request or make server support keeping connection
        # todo: always close connection
        socket = self.connect(self.address)
        metadata_json = json.dumps(metadata)
        message_out = protocol.RequestFileSubmissionBegin(metadata_json)
        message = self._get_response(message_out, socket)

        if message.msg == "operation_failure":
            raise OperationFailedError

        submission_id = message.submission_id

        total_bytes_read = 0
        part = 0

        hasher = hashlib.sha256()
        if progress_bar:
            pbar = tqdm(total=file_size, desc='Uploading slice', unit='bytes', unit_scale=True)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            total_bytes_read += len(chunk)
            hasher.update(chunk)

            self._send_chunk(chunk, submission_id, part, socket)

            if progress_bar:
                pbar.update(len(chunk))

            part += 1
        
        if progress_bar:
            pbar.close()
        checksum = hasher.hexdigest()
        message_out = protocol.RequestFileSubmissionEnd(submission_id, checksum)
        socket = self.connect(self.address)
        message = self._get_response(message_out, socket)
        
        if message.msg == "operation_failure":
            raise OperationFailedError

        if message.msg == "file_submission_end_response":
            if message.total_size != total_bytes_read:
                raise OperationFailedError
            return message.get_body()
        else:
            raise OperationFailedError(f'Unexpected message code in response: {message.msg}')

    def _send_chunk(self, data, submission_id, part, socket, max_retries=3):
        error_msg = ''
        for _ in range(max_retries):
            socket = self.connect(self.address)
            message_out = protocol.RequestSubmitPart(submission_id, part, data)

            message = self._get_response(message_out, socket)

            msg = message.msg
            if msg == 'operation_failure':
                if message.error == 'integrity_error':
                    error_msg = 'Part of file got corrupted during transfer'
                elif message.error == 'upload_not_found':
                    error_msg = 'Upload not found on the side of the server'
            elif msg == 'submit_part_response':
                if len(data) == message.part_size:
                    return
                continue
            else:
                raise OperationFailedError(f'Unexpected message code in response: {msg}')
        
        raise OperationFailedError(error_msg)

    def list_all_slices(self):
        """List all slices pushed to the compute node"""
        socket = self.connect(self.address)
        message_out = protocol.RequestAllSlices()
        message = self._get_response(message_out, socket)
        return json.loads(message.slices_json)

    def load_slice(self, name):
        """Load to memory a model slice with a given name"""
        socket = self.connect(self.address)
        message_out = protocol.RequestLoadSlice(name=name)
        message = self._get_response(message_out, socket)
        
        if message.get_message() == 'operation_failure':
            raise OperationFailedError('')
        return message.get_body()

    def clear_context(self):
        """Remove model keys and values stored in cache"""
        socket = self.connect(self.address)
        message_out = protocol.RequestClearContext()
        message = self._get_response(message_out, socket)

        if message.get_message() == 'operation_failure':
            raise OperationFailedError('')
        return message.get_body()

    def get_status(self):
        """Receive compute node readiness status and meta information about the slice"""
        socket = self.connect(self.address)
        message_out = protocol.RequestStatus()
        message = self._get_response(message_out, socket)
        return json.loads(message.status_json)

    def propagate_forward(self, tensor, shape):
        """Send a tensor to a remote node and propagate it forward through layers of the slice"""
        socket = self.connect(self.address)
        axis0, axis1 = shape
        message_out = protocol.RequestPropagateForward(axis0, axis1, tensor)

        message = self._get_response(message_out, socket)

        if message.get_message() == "operation_failure":
            raise OperationFailedError
        elif message.get_message() == "tensor_response":

            if shape[0] == message.axis0 and shape[1] == message.axis1:
                return {
                    'shape': [message.axis0, message.axis1],
                    'values': message.values
                }
            else:
                raise OperationFailedError
        else:
            raise Exception(f'Cannot handle unrecognized message')

    def _get_response(self, request, socket):
        request.send(socket)
        message_text, body = protocol.receive_message(socket)
        return protocol.restore_message(message_text, body)


class OperationFailedError(Exception):
    pass


def connect(address):
    """Connects to the remote compute node, returns a socket"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(address)
    return sock


def disconnect(sock):
    sock.close()