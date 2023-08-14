import unittest

from tests.unit.mocks import StableSocketMock, VaryingChunkSocketMock
from distllm import utils
from distllm.utils import encode_length


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


if __name__ == '__main__':
    unittest.main()
