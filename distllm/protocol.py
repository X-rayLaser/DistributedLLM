from dataclasses import dataclass
import hashlib
from typing import ClassVar
from .utils import encode_length, ByteCoder, ByteCodingError, ByteStreamParser, SocketReader


message_registry = {}


def register_message_class(cls):
    message_registry[cls.msg] = cls


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_message_class(cls)
        return cls


class Message(metaclass=Meta):
    msg = ""
    
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

    def __eq__(self, message: object) -> bool:
        return type(self) == type(message) and self.get_body() == message.get_body()

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
class RequestPropagateForward(Message):
    msg: ClassVar[str] = "propagate_forward_request"
    axis0: int
    axis1: int
    values: list


@dataclass
class ResponsePropagateForward(Message):
    msg: ClassVar[str] = "tensor_response"
    axis0: int
    axis1: int
    values: list


@dataclass
class RequestFileSubmissionBegin(Message):
    msg: ClassVar[str] = "request_file_submission_begin"
    metadata_json: str


@dataclass
class ResponseFileSubmissionBegin(Message):
    msg: ClassVar[str] = "file_submission_begin_response"
    submission_id: int


@dataclass
class RequestSubmitPart(Message):
    msg: ClassVar[str] = "request_submit_part"
    submission_id: int
    part_number: int
    data: bytes


@dataclass
class ResponseSubmitPart(Message):
    msg: ClassVar[str] = "submit_part_response"
    part_size: int


@dataclass
class RequestFileSubmissionEnd(Message):
    msg: ClassVar[str] = "request_file_submission_end"
    submission_id: int
    checksum: str


@dataclass
class ResponseFileSubmissionEnd(Message):
    msg: ClassVar[str] = "file_submission_end_response"
    file_name: str
    total_size: int


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
class ResponseWithError(Message):
    msg: ClassVar[str] = "operation_failure"
    operation: str
    error: str
    description: str


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
    message_cls = message_registry.get(message)
    if message_cls:
        return message_cls.from_body(body)
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
        elif isinstance(field, float):
            field_type = coder.encode_string('float')
            field_data = coder.encode_float(field)
        elif isinstance(field, list):
            field_type = coder.encode_string('list')
            field_data = coder.encode_list(field)
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
        elif param_type == 'float':
            value = parser.parse_float()
        elif param_type == 'list':
            value = parser.parse_list()
        elif param_type == 'NoneType':
            parser.parse_string()
            value = None
        else:
            raise ByteCodingError(f'Unknown parameter type: {param_type}')
        body[param_name] = value
    
    return message, body



class TooLongMessageStringError(Exception):
    pass
