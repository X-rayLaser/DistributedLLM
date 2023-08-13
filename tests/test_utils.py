import unittest
import sys
import struct
import hashlib
import json

sys.path.append('distllm')

from utils import ControlCenter, ModelSlice, NodeProvisioningError, Connection
import utils


class ControlCenterTests(unittest.TestCase):
    nodes_map = {
        'first_node': ('IP1', 23498),
        'second_node': ('IP2', 28931)
    }

    def test_get_status_initially(self):
        center = ControlCenter(self.nodes_map)

        status = {
            'ready': False,
            'model': None,
            'nodes': {
                'first_node': {
                    'connectivity': True,
                    'ip': 'IP1',
                    'port': 23498,
                    'status': 'brand_new',
                    'errors': [],
                    'slice': None
                },
                'second_node': {
                    'connectivity': True,
                    'ip': 'IP2',
                    'port': 28931,
                    'status': 'brand_new',
                    'errors': [],
                    'slice': None
                }
            }
        }
        self.assertEqual(status, center.get_status())

    def test_push_model_with_invalid_mapping(self):
        center = ControlCenter(self.nodes_map)
        slice1 = ModelSlice(b'Some data', 0, 20)
        slice2 = ModelSlice(b'Some data2', 21, 30)
        slice3 = ModelSlice(b'Some data2', 21, 30)

        # pass empty map
        slices = {}
        self.assertRaises(NodeProvisioningError, lambda: center.push_model('model', slices))

        # assigning slice to a node missing from the nodes map
        slices = dict(unknown_node=slice1)
        self.assertRaises(NodeProvisioningError, lambda: center.push_model('model', slices))

        # second node without a slice
        slices = dict(first_node=slice1)
        self.assertRaises(NodeProvisioningError, lambda: center.push_model('model', slices))

        # first node without a slice
        slices = dict(second_node=slice1)
        self.assertRaises(NodeProvisioningError, lambda: center.push_model('model', slices))

        # using node missing from the nodes map
        slices = dict(first_node=slice1, second_node=slice2, unknown_node=slice3)
        self.assertRaises(NodeProvisioningError, lambda: center.push_model('model', slices))

    def test_push_model_and_get_status(self):
        center = ControlCenter(self.nodes_map)
        slice1 = ModelSlice(b'First slice data', 0, 20)
        slice2 = ModelSlice(b'Second slice data', 21, 31)

        slices = dict(first_node=slice1, second_node=slice2)
        center.push_model('model', slices)

        status = center.get_status()

        expected_status = {
            'ready': True,
            'model': {
                'pseudoname': 'model',
                'family': None,
                'class': None,
                'size': None
            },
            'nodes': {
                'first_node': {
                    'connectivity': True,
                    'ip': 'IP1',
                    'port': 23498,
                    'status': 'up',
                    'errors': [],
                    'slice': [0, 20]
                },
                'second_node': {
                    'connectivity': True,
                    'ip': 'IP2',
                    'port': 28931,
                    'status': 'up',
                    'errors': [],
                    'slice': [21, 31]
                }
            }
        }

        self.assertEqual(expected_status, status)


class StableSocketMock:
    def __init__(self) -> None:
        self.data = bytes()
        self.step = 1
        self.idx = 0

    def recv(self, max_size):
        chunk = self.data[self.idx:self.idx + self.step]
        self.idx += self.step
        return chunk

    def sendall(self, buffer):
        self.data = buffer
        self.idx = 0


class VaryingChunkSocketMock(StableSocketMock):
    def __init__(self):
        super().__init__()
        self.num_reads = 0

    def recv(self, max_size):
        self.step = self.num_reads % 4
        self.num_reads += 1
        return super().recv(max_size)


class SimpleServerSocketMock(StableSocketMock):
    def __init__(self) -> None:
        super().__init__()
        self.message = None

    def sendall(self, buffer):
        """Pretend that this code executes by server which store response in data attribute"""
        msg = self.message.get_message()
        body = self.message.get_body()
        self.data = utils.encode_message(msg, body)

    def set_reply_message(self, message):
        """Sets response that client will receive"""
        self.message = message


class ComplexServerSocketMock(StableSocketMock):
    def __init__(self):
        super().__init__()
        self.responses = {}

    def sendall(self, buffer):
        super().sendall(buffer)  # store data in the instance attributes

        # use subroutines to read data from socket (this instance) and decode client message
        msg, body = utils.receive_message(self)
        message = utils.restore_message(msg, body)
        message_text = message.get_message()

        # use appropriate response message with mocked body from the test code
        if message_text == "slices_request":
            message_out_class = utils.JsonResponseWithSlices
        elif message_text == "status_request":
            message_out_class = utils.JsonResponseWithStatus
        elif message_text == "load_slice_request":
            message_out_class = utils.JsonResponseWithLoadedSlice
        else:
            raise Exception('Unrecognized message')
        
        body_out = self.responses[message_text]
        message_out = message_out_class(**body_out)
        
        self.data = message_out.encode()
        self.idx = 0

    def set_reply_body(self, msg, body):
        self.responses[msg] = body


class ConnectionWithMockedServerTests(unittest.TestCase):
    address = ('name', 'some_ip', 22304)

    def setUp(self):
        self.connection = Connection(self.address)
        self.socket = ComplexServerSocketMock()
        self.connection.connect = lambda _ : self.socket

    def test_get_status_for_node_with_a_slice(self):
        excepted_status = {
            'model': None,
            'connectivity': True,
            'ip': 'some_ip',
            'port': 22304,
            'status': 'up',
            'errors': [],
            'slice': [4, 23]
        }

        status_json = json.dumps(excepted_status)
        body = dict(status_json=status_json)
        self.socket.set_reply_body("status_request", body)

        status = self.connection.get_status()

        self.assertEqual(excepted_status, status)

    def test_list_all_slices_gives_empty_list(self):
        slices_json = json.dumps([])
        self.socket.set_reply_body("slices_request", body=dict(slices_json=slices_json))

        self.assertEqual([], self.connection.list_all_slices())

    def test_list_all_slices(self):
        expected_slices = [{
            'model': 'llama_v1',
            'layer_from': 0,
            'layer_to': 12
        }, {
            'model': 'falcon',
            'layer_from': 12,
            'layer_to': 28
        }]

        slices_json = json.dumps(expected_slices)
        self.socket.set_reply_body("slices_request", body=dict(slices_json=slices_json))

        self.assertEqual(expected_slices, self.connection.list_all_slices())

    def test_load_slice(self):
        expected = {
            'name': 'funky',
            'model': 'falcon'
        }

        self.socket.set_reply_body("load_slice_request", body=expected)
        self.assertEqual(expected, self.connection.load_slice('funky'))


# todo: test sending and receiving of each Message subclasses
class ImplementedMessagesTests(unittest.TestCase):
    def test(self):
        pass


class StableSocketTests(unittest.TestCase):
    def setUp(self):
        self.socket = StableSocketMock()

    def test_receive_0_bytes(self):
        data = b''
        
        self.socket.data = encode_length(len(data)) + data
        received_data = utils.receive_data(self.socket)
        self.assertEqual(b'', received_data)

    def test_receive_1_byte(self):
        data = bytes([134])
        self.socket.data = encode_length(len(data)) + data

        received_data = utils.receive_data(self.socket)
        self.assertEqual(bytes(data), received_data)

    def test_receive_2_bytes(self):
        data = bytes([23, 134])
        self.socket.data = encode_length(len(data)) + data
        self.step = 3
        received_data = utils.receive_data(self.socket)
        self.assertEqual(bytes(data), received_data)

    def test_receive_few_bytes(self):
        data = bytes([1, 2, 3, 4, 5, 6, 7])
        self.socket.data = encode_length(len(data)) + data
        self.step = 3
        received_data = utils.receive_data(self.socket)
        self.assertEqual(bytes(data), received_data)


class UnstableSocketTests(unittest.TestCase):
    def test_receive_few_bytes(self):
        socket = VaryingChunkSocketMock()
        data = bytes([155, 23, 94, 54, 213, 92, 188, 211, 153, 200])
        socket.data = encode_length(len(data)) + data
        received_data = utils.receive_data(socket)
        self.assertEqual(bytes(data), received_data)

    def test_receive_consequitve_data_streams(self):
        socket = VaryingChunkSocketMock()
        data1 = bytes([155, 23, 94, 54, 213, 92, 188, 211, 153, 200])
        data2 = bytes([1, 2, 3, 4, 5, 6, 7])
        socket.data = encode_length(len(data1)) + data1 + encode_length(len(data2)) + data2

        reader = utils.SocketReader(socket)
        first_portion = reader.receive_data()
        second_portion = reader.receive_data()
        total = first_portion + second_portion
        self.assertEqual(data1 + data2, total)


class SendAndReceiveMessageTests(unittest.TestCase):
    def setUp(self):
        self.socket = StableSocketMock()

    def test_send_too_long_message(self):
        message = 'm' * 31
        self.assertRaises(utils.TooLongMessageStringError,
                          lambda: utils.send_message(self.socket, message, {}))

    def test_send_empty_message(self):
        utils.send_message(self.socket, '', {})
        message, body = utils.receive_message(self.socket)
        self.assertEqual('', message)
        self.assertEqual({}, body)

    def test_send_blank_message(self):
        # surronding spaces should be ignored
        utils.send_message(self.socket, '         ', {})
        message, body = utils.receive_message(self.socket)
        self.assertEqual('', message)
        self.assertEqual({}, body)

    def test_send_message_without_body(self):
        original = ' hello, world!   '
        utils.send_message(self.socket, original, {})
        message, body = utils.receive_message(self.socket)
        self.assertEqual(original.strip(), message)
        self.assertEqual({}, body)

    def test_send_message_with_nones(self):
        original = 'good luck decoding this!'
        body = dict(foo='some string', bar=None, param=23,
                    another_none=None, yet_another=None, non_none="None")
        utils.send_message(self.socket, original, body)
        message, received_body = utils.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(body, received_body)

    def test_send_message_with_body(self):
        original = 'add numbers'
        body = dict(term1=3284, term2=102)
        utils.send_message(self.socket, original, body)
        message, received_body = utils.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(body, received_body)

    def test_send_message_with_diverse_payload(self):
        original = 'do something'
        body = dict(foo="some text", bar=32, extra_data=bytes([12, 39, 104, 210]))
        utils.send_message(self.socket, original, body)
        message, received_body = utils.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(body, received_body)


class DataTransmissionTests(unittest.TestCase):
    def test_send_empty_message(self):
        socket = StableSocketMock()
        utils.send_message(socket, '', {})

        data = self._encode_message('')
        data += utils.ByteCoder().encode_int(0)
        expected_data = self._get_expected_data(data)
        self.assertEqual(expected_data, socket.data)

    def test_send_message_without_body(self):
        socket = StableSocketMock()
        original_msg = 'hello world'
        utils.send_message(socket, original_msg, {})

        data = self._encode_message(original_msg)
        data += utils.ByteCoder().encode_int(0)
        expected_data = self._get_expected_data(data)
        self.assertEqual(expected_data, socket.data)

    def test_send_message_with_body(self):
        socket = StableSocketMock()
        original_msg = 'hello world'
        body = dict(foo="foo", bar=434)
        utils.send_message(socket, original_msg, body)

        coder = utils.ByteCoder()
        message_binary = self._encode_message(original_msg)

        payload = message_binary + coder.encode_int(len(body))
        payload += coder.encode_string('foo') + coder.encode_string('str') + coder.encode_string('foo')
        payload += coder.encode_string('bar') + coder.encode_string('int') + coder.encode_int(434)
        expected_data = self._get_expected_data(payload)
        self.assertEqual(expected_data, socket.data)

    def _encode_message(self, message):
        s = message + ' ' * (utils.MAX_MESSAGE_SIZE - len(message))
        return bytes(s, encoding='ascii')

    def _get_expected_data(self, payload):
        checksum = hashlib.sha256(payload).hexdigest().encode('ascii')
        size_bytes = encode_length(len(checksum) + len(payload))
        return size_bytes + checksum + payload


class ByteCoderTests(unittest.TestCase):
    def setUp(self):
        self.coder = utils.ByteCoder()

    def test_on_integers(self):
        value = 0
        int_bytes = self.coder.encode_int(value)
        new_offset = 4
        expected_value = (value, new_offset)
        self.assertEqual(expected_value, self.coder.decode_int(int_bytes))
        self.assertEqual(expected_value, self.coder.decode_int(int_bytes + b'ignored_bytes'))

        value = 392384
        expected_value = (value, new_offset)
        encoded_value = self.coder.encode_int(value)
        self.assertEqual(expected_value, self.coder.decode_int(encoded_value))
        self.assertEqual(expected_value, self.coder.decode_int(encoded_value + b'ignored_bytes'))

        value = 2**34  # won't fit in 4 bytes
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.encode_int(value))

        # not enough bytes
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.decode_int(b'aef'))

    def test_on_strings(self):
        # decoding string that is smaller than its specified length
        length = 5
        binary_str = self.coder.encode_int(length) + b'abcd'
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.decode_string(binary_str))

        # decode corrupted string
        binary_str = self.coder.encode_int(length) + bytes([233, 121, 212, 120, 123])
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.decode_string(binary_str))

        # encode and decode ascii string
        s = 'Hello, world'

        binary_str = self.coder.encode_string(s)
        expected_value = (s, 4 + len(s))
        self.assertEqual(expected_value, self.coder.decode_string(binary_str))

        self.assertEqual(expected_value, self.coder.decode_string(binary_str + b'extrabytes'))

    def test_on_binary_data(self):
        # not enough bytes to decode blob of a given size
        blob = bytes([29, 10, 32, 255])
        encoded_blob = self.coder.encode_int(5)
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.decode_blob(encoded_blob))

        blob = bytes()
        encoded_blob = self.coder.encode_blob(blob)
        
        padding = b'extra bytes'
        self.assertEqual((b'', 4), self.coder.decode_blob(encoded_blob))
        self.assertEqual((b'', 4), self.coder.decode_blob(encoded_blob + padding))

        blob = bytes([123, 212, 82, 0, 255, 12])
        encoded_blob = self.coder.encode_blob(blob)
        expected_value = (blob, len(blob) + 4)
        self.assertEqual(expected_value, self.coder.decode_blob(encoded_blob))
        self.assertEqual(expected_value, self.coder.decode_blob(encoded_blob + padding))



def encode_length(length):
    return struct.pack('i', length)


if __name__ == '__main__':
    unittest.main()
