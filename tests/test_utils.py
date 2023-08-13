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
        self.errors = {}
        self.responses = {}

    def sendall(self, buffer):
        super().sendall(buffer)  # store data in the instance attributes

        # use subroutines to read data from socket (this instance) and decode client message
        msg, body = utils.receive_message(self)
        message = utils.restore_message(msg, body)
        message_text = message.get_message()

        # use appropriate response message with mocked body from the test code
        error_body_out = self.errors.get(message_text)

        response_classes = {
            'slices_request': utils.JsonResponseWithSlices,
            'status_request': utils.JsonResponseWithStatus,
            'load_slice_request': utils.JsonResponseWithLoadedSlice
        }

        error_response_classes = {
            'load_slice_request': utils.ResponseWithError
        }

        if error_body_out:
            message_out_class = error_response_classes[message_text]
            message_out = message_out_class(**error_body_out)
        else:
            try:
                message_out_class = response_classes[message_text]
            except KeyError:
                raise Exception(f'No response for message {message_text}')
            body_out = self.responses[message_text]
            message_out = message_out_class(**body_out)
        
        self.data = message_out.encode()
        self.idx = 0

    def set_error_body(self, msg, body):
        self.errors[msg] = body

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

    def test_load_slice_unsuccessful(self):
        expected = {
            'operation': 'load_slice',
            'error': 'Brief error',
            'description': 'Details'
        }
        self.socket.set_error_body("load_slice_request", body=expected)
        self.assertRaises(utils.OperationFailedError,
                          lambda: self.connection.load_slice('funky'))


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
        body = dict(foo="some text",
                    bar=32,
                    extra_data=bytes([12, 39, 104, 210]))
        utils.send_message(self.socket, original, body)
        message, received_body = utils.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(body, received_body)

    def test_send_message_with_float_param(self):
        original = 'do something'
        body = dict(floating_point_number=3284.2149)
        utils.send_message(self.socket, original, body)
        message, received_body = utils.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(set(body.keys()), set(received_body.keys()))
        self.assertEqual(len(body), len(received_body))
        self.assertAlmostEqual(body["floating_point_number"],
                               received_body["floating_point_number"], places=3)

    def test_send_message_with_list_param(self):
        original = 'do something'
        alist = [42.9, -21.385, 8032.104, 734.0]
        body = dict(alist=alist)
        utils.send_message(self.socket, original, body)
        message, received_body = utils.receive_message(self.socket)

        self.assertEqual(original, message)

        self.assertEqual(len(alist), len(received_body["alist"]))
        for i in range(len(alist)):
            self.assertAlmostEqual(alist[i], received_body["alist"][i], places=3)


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

    def test_on_floats(self):
        encoded_float = self.coder.encode_float(0)
        new_offset = 4
        expected_value = (0, new_offset)
        self.assertEqual(expected_value, self.coder.decode_float(encoded_float))
        self.assertEqual(expected_value, self.coder.decode_float(encoded_float + b'ignored_bytes'))

        value = 432.84302
        encoded_value = self.coder.encode_float(value)
        decoded_value, new_pos = self.coder.decode_float(encoded_value + b'ignored_bytes')

        self.assertEqual(new_offset, new_pos)
        self.assertAlmostEqual(value, decoded_value, places=4)

        # not enough bytes
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.decode_float(b'aef'))

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

    def test_on_lists_of_floats(self):
        # decoding list that is smaller than its specified size
        size = 3
        list_bytes = self.coder.encode_int(size) + b'abcdefgh'
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.decode_list(list_bytes))

        # number of bytes in the payload is not multiple of 4
        list_bytes = self.coder.encode_int(size) + b'abcdefghij'
        self.assertRaises(utils.ByteCodingError, lambda: self.coder.decode_list(list_bytes))

        alist = [0.24, 832., 142.5, 0.0, -241.90]
        encoded_list = self.coder.encode_list(alist)
        decoded_list, new_pos = self.coder.decode_list(encoded_list + b'Bytes to ignore')
        self.assertEqual(len(alist) * 4 + 4, new_pos)

        self.assertEqual(len(alist), len(decoded_list))
        for i in range(len(alist)):
            self.assertAlmostEqual(alist[i], decoded_list[i], places=2)

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
