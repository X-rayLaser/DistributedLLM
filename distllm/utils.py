from dataclasses import dataclass
import hashlib
import struct
import json
from typing import ClassVar


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
        message_out = RequestAllSlices()
        message_out.send(socket)

        message_text, body = receive_message(socket)
        message = restore_message(message_text, body)
        return json.loads(message.slices_json)

    def load_slice(self, name):
        """Load to memory a model slice with a given name"""
        socket = self.connect(self.address)
        message_out = RequestLoadSlice(name=name)
        message_out.send(socket)

        message_text, body = receive_message(socket)
        message = restore_message(message_text, body)
        return message.get_body()

    def get_status(self):
        """Receive compute node readiness status and meta information about the slice"""
        socket = self.connect(self.address)
        message_out = RequestStatus()
        message_out.send(socket)
        message_text, body = receive_message(socket)
        message = restore_message(message_text, body)
        return json.loads(message.status_json)

    def propagate_forward(self, tensor):
        """Send a tensor to a remote node and propagate it forward through layers of the slice"""


@dataclass
class Message:
    msg: ClassVar[str]
    
    def send(self, socket):
        body = self.get_body()
        send_message(socket, self.msg, body)

    def encode(self):
        body = self.get_body()
        return encode_message(self.msg, body)

    def get_message(self):
        return self.msg

    def get_body(self):
        return {name: value for name, value in self.__dict__.items() if name != 'msg'}

    @classmethod
    def from_body(cls, body):
        return cls(**body)


@dataclass
class RequestAllSlices(Message):
    msg: ClassVar[str] = "slices_request"


@dataclass
class RequestStatus(Message):
    msg: ClassVar[str] = "status_request"


@dataclass
class RequestLoadSlice(Message):
    msg: ClassVar[str] = "load_slice_request"
    name: str


@dataclass
class JsonResponseWithStatus(Message):
    msg: ClassVar[str] = "status_response"
    status_json: str

    @classmethod
    def from_body(cls, body):
        return cls(**body)


@dataclass
class JsonResponseWithSlices(Message):
    msg: ClassVar[str] = "slices_list_response"
    slices_json: str


@dataclass
class JsonResponseWithLoadedSlice(Message):
    msg: ClassVar[str] = "loaded_slice_response"
    name: str
    model: str


@dataclass
class SliceSubmissionBegin(Message):
    model: str
    layer_from: int
    layer_to: int

    @classmethod
    def from_body(cls, body):
        cls()


@dataclass
class SliceSubmissionPart(Message):
    submission_id: int
    part_id: int
    data: bytes


@dataclass
class SliceSubmissionEnd(Message):
    submission_id: int
    sha256_digest: str


def restore_message(message, body):
    if message == 'status_response':
        return JsonResponseWithStatus(status_json=body['status_json'])
    elif message == 'slices_list_response':
        return JsonResponseWithSlices(slices_json=body['slices_json'])
    elif message == 'slices_request':
        return RequestAllSlices()
    elif message == 'status_request':
        return RequestStatus()
    elif message == 'load_slice_request':
        return RequestLoadSlice.from_body(body)
    elif message == 'loaded_slice_response':
        return JsonResponseWithLoadedSlice.from_body(body)
    else:
        raise Exception(f'Unrecognized message {message}')


MAX_MESSAGE_SIZE = 30


def send_message(socket, message, body):
    all_data = encode_message(message, body)
    socket.sendall(all_data)


def encode_message(message, body):
    payload = bytearray()

    if len(message) > MAX_MESSAGE_SIZE:
        raise TooLongMessageStringError('')

    coder = ByteCoder()

    padding = ' ' * (MAX_MESSAGE_SIZE - len(message))

    payload.extend((message + padding).encode('ascii'))

    num_params = len(body)
    payload.extend(coder.encode_int(num_params))
    for name, field in body.items():
        payload.extend(coder.encode_string(name))
        if isinstance(field, bytes):
            field_type = coder.encode_string('bytes')
            field_data = coder.encode_blob(field)
        elif isinstance(field, str):
            field_type = coder.encode_string('str')
            field_data = coder.encode_string(field)
        elif isinstance(field, int):
            field_type = coder.encode_string('int')
            field_data = coder.encode_int(field)
        elif field is None:
            field_type = coder.encode_string('NoneType')
            field_data = coder.encode_string('None')
        else:
            raise Exception(f'Unsupport data type: {field}')
        payload.extend(field_type)
        payload.extend(field_data)
    
    digest_data = hashlib.sha256(payload).hexdigest().encode('ascii')
    total_size = len(digest_data) + len(payload)

    all_data = encode_length(total_size) + digest_data + bytes(payload)
    return all_data


class TooLongMessageStringError(Exception):
    pass


def receive_message(socket):
    reader = SocketReader(socket)
    data = reader.receive_data()
    payload = data[64:]
    hex_digest = data[:64].decode('ascii')
    if hashlib.sha256(payload).hexdigest() != hex_digest:
        raise Exception('Data integrity error. Hashes do not match')

    message = payload[:MAX_MESSAGE_SIZE].decode('ascii').strip()
    body_data = payload[MAX_MESSAGE_SIZE:]

    parser = ByteStreamParser(body_data)
    num_params = parser.parse_int()

    body = {}
    for i in range(num_params):
        param_name = parser.parse_string()
        param_type = parser.parse_string()
        if param_type == 'bytes':
            value = parser.parse_blob()
        elif param_type == 'str':
            value = parser.parse_string()
        elif param_type == 'int':
            value = parser.parse_int()
        elif param_type == 'NoneType':
            parser.parse_string()
            value = None
        else:
            raise ByteCodingError(f'Unknown parameter type: {param_type}')
        body[param_name] = value
    
    return message, body


class ByteStreamParser:
    def __init__(self, byte_stream):
        self.stream = byte_stream
        self.pos = 0
        self.coder = ByteCoder()

    def parse_int(self):
        return self._parse(self.coder.decode_int)

    def parse_string(self):
        return self._parse(self.coder.decode_string)

    def parse_blob(self):
        return self._parse(self.coder.decode_blob)

    def _parse(self, decode_func):
        value, new_pos = decode_func(self.stream[self.pos:])
        self.pos += new_pos
        return value


class ByteCoder:
    def encode_int(self, value):
        return self._encode_int('i', value)

    def decode_int(self, value_bytes):
        if len(value_bytes) < 4:
            raise ByteCodingError
        
        new_byte_position = 4
        return self._decode_int('i', value_bytes[:4]), new_byte_position

    def encode_string(self, s):
        utf_enc = s.encode('utf-8')
        length = self.encode_int(len(utf_enc))
        return length + utf_enc

    def decode_string(self, str_bytes):
        length, new_pos = self.decode_int(str_bytes[:4])

        if len(str_bytes) - 4 < length:
            raise ByteCodingError

        utf_enc = str_bytes[new_pos:new_pos + length]

        try:
            s = utf_enc.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ByteCodingError(str(e))

        return s, new_pos + length

    def encode_blob(self, blob):
        length = self.encode_int(len(blob))
        return length + blob

    def decode_blob(self, encoded_blob):
        length, payload_offset = self.decode_int(encoded_blob[:4])

        new_pos = payload_offset + length

        payload_size = len(encoded_blob) - payload_offset
        if payload_size < length:
            raise ByteCodingError

        return encoded_blob[payload_offset:new_pos], new_pos

    def _encode_int(self, format, value):
        try:
            return struct.pack(format, value)
        except struct.error as e:
            raise ByteCodingError(str(e))

    def _decode_int(self, format, value_bytes):
        try:
            return struct.unpack(format, value_bytes)[0]
        except struct.error as e:
            raise ByteCodingError(str(e))


class ByteCodingError(Exception):
    pass


def encode_length(length):
    return struct.pack('i', length)


def connect(address):
    """Connects to the remote compute node, returns a socket"""


def encode_propagate_forward(tensor):
    """Convert tensor into bytes to send"""


def encode_push_slice_begin(name, layer_range):
    """Encode command to begin sending slice"""


def encode_push_slice_part(submission_id, part, sha256_digest, data):
    """Encode command part of a model slice"""


def encode_push_slice_end(submission_id, sha256_digest):
    """Encode command to finish sending slice"""


# todo: more commands, test this whole thing


def receive_data(socket, chunk_size=1024):
    """
    Read all data from a socket by chunks.

    Expects first 4 bytes to contain number of bytes to read in the remaining stream.
    """
    stream = SocketReader(socket, chunk_size)
    return stream.receive_data()


class SocketReader:
    def __init__(self, socket, chunk_size=1024):
        self.socket = socket
        self.buffer = b''
        self.chunk_size = chunk_size

    def receive_data(self):
        size_bytes = self.receive_bytes(4)
        num_bytes = struct.unpack('i', size_bytes)[0]

        return self.receive_bytes(num_bytes)

    def receive_bytes(self, n):
        all_received_bytes = bytearray()
        while True:
            received = self.receive_chunk()
            all_received_bytes.extend(received)
            if len(all_received_bytes) >= n:
                break
        
        all_received_bytes = bytes(all_received_bytes)
        extra_tail = len(all_received_bytes) - n

        res = all_received_bytes[:n]
        extra_tail = all_received_bytes[n:]

        self.buffer = extra_tail

        return res

    def receive_chunk(self):
        if self.buffer:
            res = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]
            return res
        return self.socket.recv(self.chunk_size)


def parse_bytes(byte_seq):
    """
    Parse a stream of bytes
    """


def parse_command(byte_seq):
    """
    Parse a command

    Returns a tuple containing a command and an offset for the payload
    """


def parse_tensor(byte_seq):
    """
    Parse tensor data and its dimensions, verify data integrity

    Format: # dimensions (1 byte), 4 bytes per dimension, 16 bytes SHA256 checksum, tensor data
    """


def parse_slice_init(byte_seq):
    """
    Denotes the upcoming submission of segments of model slice.
    Contains information about slice submission id and meta data (name length, name, layers range (4 bytes))
    """


def parse_slice_finilize(byte_seq):
    """
    Denotes that all segments of slice were submitted.
    Contains the submission id, final checksum of the whole slice
    """


def parse_slice_part(byte_seq):
    """
    Parse part of a model slice, verify data integrity

    Format: submission id, part number (2 bytes), 16 bytes SHA256 checksum, data
    """


def parse_slice_to_load(byte_seq):
    """
    Format: name length in bytes (2 bytes), a name given to the slice
    """