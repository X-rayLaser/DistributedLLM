import unittest
import hashlib
from distllm import protocol, utils
from distllm.utils import encode_length
from mocks import StableSocketMock, VaryingChunkSocketMock


class SendAndReceiveMessageTests(unittest.TestCase):
    def setUp(self):
        self.socket = StableSocketMock()

    def test_send_too_long_message(self):
        message = 'm' * 31
        self.assertRaises(protocol.TooLongMessageStringError,
                          lambda: protocol.send_message(self.socket, message, {}))

    def test_send_empty_message(self):
        protocol.send_message(self.socket, '', {})
        message, body = protocol.receive_message(self.socket)
        self.assertEqual('', message)
        self.assertEqual({}, body)

    def test_send_blank_message(self):
        # surronding spaces should be ignored
        protocol.send_message(self.socket, '         ', {})
        message, body = protocol.receive_message(self.socket)
        self.assertEqual('', message)
        self.assertEqual({}, body)

    def test_send_message_without_body(self):
        original = ' hello, world!   '
        protocol.send_message(self.socket, original, {})
        message, body = protocol.receive_message(self.socket)
        self.assertEqual(original.strip(), message)
        self.assertEqual({}, body)

    def test_send_message_with_nones(self):
        original = 'good luck decoding this!'
        body = dict(foo='some string', bar=None, param=23,
                    another_none=None, yet_another=None, non_none="None")
        protocol.send_message(self.socket, original, body)
        message, received_body = protocol.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(body, received_body)

    def test_send_message_with_body(self):
        original = 'add numbers'
        body = dict(term1=3284, term2=102)
        protocol.send_message(self.socket, original, body)
        message, received_body = protocol.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(body, received_body)

    def test_send_message_with_diverse_payload(self):
        original = 'do something'
        body = dict(foo="some text",
                    bar=32,
                    extra_data=bytes([12, 39, 104, 210]))
        protocol.send_message(self.socket, original, body)
        message, received_body = protocol.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(body, received_body)

    def test_send_message_with_float_param(self):
        original = 'do something'
        body = dict(floating_point_number=3284.2149)
        protocol.send_message(self.socket, original, body)
        message, received_body = protocol.receive_message(self.socket)
        self.assertEqual(original, message)
        self.assertEqual(set(body.keys()), set(received_body.keys()))
        self.assertEqual(len(body), len(received_body))
        self.assertAlmostEqual(body["floating_point_number"],
                               received_body["floating_point_number"], places=3)

    def test_send_message_with_list_param(self):
        original = 'do something'
        alist = [42.9, -21.385, 8032.104, 734.0]
        body = dict(alist=alist)
        protocol.send_message(self.socket, original, body)
        message, received_body = protocol.receive_message(self.socket)

        self.assertEqual(original, message)

        self.assertEqual(len(alist), len(received_body["alist"]))
        for i in range(len(alist)):
            self.assertAlmostEqual(alist[i], received_body["alist"][i], places=3)



class DataTransmissionTests(unittest.TestCase):
    def test_send_empty_message(self):
        socket = StableSocketMock()
        protocol.send_message(socket, '', {})

        data = self._encode_message('')
        data += utils.ByteCoder().encode_int(0)
        expected_data = self._get_expected_data(data)
        self.assertEqual(expected_data, socket.data)

    def test_send_message_without_body(self):
        socket = StableSocketMock()
        original_msg = 'hello world'
        protocol.send_message(socket, original_msg, {})

        data = self._encode_message(original_msg)
        data += utils.ByteCoder().encode_int(0)
        expected_data = self._get_expected_data(data)
        self.assertEqual(expected_data, socket.data)

    def test_send_message_with_body(self):
        socket = StableSocketMock()
        original_msg = 'hello world'
        body = dict(foo="foo", bar=434)
        protocol.send_message(socket, original_msg, body)

        coder = utils.ByteCoder()
        message_binary = self._encode_message(original_msg)

        payload = message_binary + coder.encode_int(len(body))
        payload += coder.encode_string('foo') + coder.encode_string('str') + coder.encode_string('foo')
        payload += coder.encode_string('bar') + coder.encode_string('int') + coder.encode_int(434)
        expected_data = self._get_expected_data(payload)
        self.assertEqual(expected_data, socket.data)

    def _encode_message(self, message):
        s = message + ' ' * (protocol.MAX_MESSAGE_SIZE - len(message))
        return bytes(s, encoding='ascii')

    def _get_expected_data(self, payload):
        checksum = hashlib.sha256(payload).hexdigest().encode('ascii')
        size_bytes = encode_length(len(checksum) + len(payload))
        return size_bytes + checksum + payload

