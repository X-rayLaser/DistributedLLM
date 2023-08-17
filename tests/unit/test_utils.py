from io import UnsupportedOperation
import os
import unittest
from distllm.utils import FakeFileSystemBackend

from tests.unit.mocks import StableSocketMock, VaryingChunkSocketMock
from distllm import utils
from distllm.utils import FakeFileTree, encode_length


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


class FakeFileTreeTests(unittest.TestCase):
    def test_empty_tree(self):
        tree = FakeFileTree()
        self.assertFalse(tree.exists('foo'))
        self.assertEqual([], tree.list_files())
        self.assertEqual([], tree.list_dirs())

    def test_adding_a_file_into_missing_folder(self):
        tree = FakeFileTree()
        path = os.path.join("foo", "bar.txt")
        self.assertRaises(FileNotFoundError, lambda: tree.add_file(path))
        self.assertFalse(tree.exists(path))

        self.assertEqual([], tree.list_files())
        self.assertEqual([], tree.list_dirs())

    def test_putting_a_file_under_the_root(self):
        tree = FakeFileTree('')

        self.assertRaises(Exception, lambda: tree.add_file(''))

        tree.add_file("file_name")
        self.assertEqual(["file_name"], tree.list_files())
        self.assertEqual([], tree.list_dirs())

    def test_adding_folder_under_root(self):
        tree = FakeFileTree('/')

        self.assertRaises(Exception, lambda: tree.make_dirs(''))

        tree.make_dirs(os.path.join("foo"))
        self.assertEqual(["/foo"], tree.list_dirs())
        self.assertTrue(tree.exists("/foo"))

    def test_adding_nested_folders(self):
        tree = FakeFileTree('')

        dir1 = "foo"
        dir2 = os.path.join("foo", "bar")
        dir3 = os.path.join("foo", "notbar")
        dir4 = os.path.join("deep", "nested", "folder")
        dir5 = os.path.join("deep", "dir")
        tree.make_dirs("foo")

        tree.make_dirs(dir2)
        self.assertEqual([dir1, dir2], tree.list_dirs())

        tree.make_dirs(dir3)
        self.assertEqual([dir1, dir2, dir3], tree.list_dirs())

        tree.make_dirs(dir4)
        expected = [dir1, dir2, dir3, "deep", os.path.join("deep", "nested"),
                    os.path.join("deep", "nested", "folder")]
        self.assertEqual(expected, tree.list_dirs())

        tree.make_dirs(dir5)

        expected += [os.path.join("deep", "dir")]
        self.assertEqual(expected, tree.list_dirs())

    def test_cannot_create_directory_if_there_already_exists_a_file(self):
        tree = FakeFileTree('')

        tree.add_file('second_file')
        path = os.path.join('second_file', 'subdir')
        self.assertRaises(FileExistsError, lambda: tree.make_dirs(path))

        self.assertFalse(tree.exists(path))
        self.assertEqual(['second_file'], tree.list_files())
        self.assertEqual([], tree.list_dirs())

    def test_cannot_create_file_in_place_of_directory(self):
        tree = FakeFileTree('')
        tree.make_dirs(os.path.join("foo", "bar"))
        self.assertRaises(FileExistsError,
                          lambda: tree.add_file('foo'))
        self.assertRaises(FileExistsError,
                          lambda: tree.add_file(os.path.join('foo', 'bar')))

    def test_cannot_create_the_same_file_twice(self):
        def make_assertions(file_tree):
            file_tree.make_dirs(os.path.join("foo", "bar"))

            file_tree.add_file(os.path.join('foo', 'bar', 'file.txt'))
            self.assertRaises(FileExistsError,
                            lambda: file_tree.add_file(os.path.join('foo', 'bar', 'file.txt')))

        make_assertions(FakeFileTree(''))
        make_assertions(FakeFileTree('/'))
        make_assertions(FakeFileTree())

    def test_cannot_create_the_same_folder_twice(self):
        def make_assertions(file_tree):
            file_tree.make_dirs(os.path.join("foo", "bar"))

            self.assertTrue(file_tree.exists(os.path.join("foo", "bar")))
            self.assertRaises(FileExistsError,
                            lambda: file_tree.make_dirs(os.path.join("foo", "bar")))

        make_assertions(FakeFileTree(''))
        make_assertions(FakeFileTree('/'))
        make_assertions(FakeFileTree())

    def test_cannot_write_or_read_non_existing_file(self):
        tree = FakeFileTree('/')
        self.assertRaises(FileNotFoundError, lambda: tree.write_to_file('foobar.txt', b'data'))
        self.assertRaises(FileNotFoundError, lambda: tree.read_file('foobar.txt'))

        tree.make_dirs("foo")
        path = os.path.join('foo', 'foobar.txt')
        self.assertRaises(FileNotFoundError, lambda: tree.write_to_file(path, b'data'))
        self.assertRaises(FileNotFoundError, lambda: tree.read_file(path))

        # cannot read or write to directory
        self.assertRaises(FileNotFoundError, lambda: tree.write_to_file('foo', b'data'))
        self.assertRaises(FileNotFoundError, lambda: tree.read_file('foo'))

        path = os.path.join("foo", "bar")
        tree.make_dirs(path)
        self.assertRaises(FileNotFoundError, lambda: tree.write_to_file(path, b'data'))
        self.assertRaises(FileNotFoundError, lambda: tree.read_file(path))

    def test_creating_directory_and_adding_files(self):
        tree = FakeFileTree('')
        tree.add_file('first_file')
        tree.add_file('second_file')
        tree.make_dirs(os.path.join('somedir', 'subdir'))
        tree.add_file(os.path.join('somedir', 'third_file'))
        tree.add_file(os.path.join('somedir', 'subdir', 'fourth_file'))

        expected_files = ['first_file', 'second_file', os.path.join('somedir', 'third_file'),
                          os.path.join('somedir', 'subdir', 'fourth_file')]

        self.assertSequenceEqual(set(expected_files), set(tree.list_files()))

    def test_making_directories_and_files_under_different_root(self):
        tree = FakeFileTree('disk_root')
        tree.add_file('first_file')
        tree.make_dirs(os.path.join('somedir', 'subdir'))
        tree.add_file(os.path.join('somedir', 'subdir', 'second_file'))

        expected = [os.path.join('disk_root', 'first_file'),
                     os.path.join('disk_root', 'somedir', 'subdir', 'second_file')]

        self.assertSequenceEqual(set(expected), set(tree.list_files()))

    def test_existence_of_folders(self):
        tree = FakeFileTree('')

        tree.make_dirs(os.path.join("foo", "bar"))

        self.assertTrue(tree.exists("foo"))
        self.assertTrue(tree.exists(os.path.join("foo", "bar")))

        self.assertFalse(tree.exists("bar"))

    def test_writing_and_reading_files(self):
        tree = FakeFileTree('')

        tree.make_dirs(os.path.join("foo", "bar"))

        file_path = os.path.join("foo", "bar", "myfile")
        tree.add_file(file_path)

        tree.write_to_file(file_path, b'hello,')
        self.assertEqual(b'hello,', tree.read_file(file_path))
        tree.write_to_file(file_path, b'world!')
        self.assertEqual(b'world!', tree.read_file(file_path))


class FakeFileSystemTests(unittest.TestCase):
    def test_exceptions(self):
        fs = FakeFileSystemBackend()
        self.assertRaises(FileNotFoundError, lambda: fs.open_file("some_file", "r"))
        self.assertRaises(FileNotFoundError,
                          lambda: fs.open_file(os.path.join("some_dir", "somefile"), "w"))

        path = os.path.join("some_dir", "subdir")
        fs.make_dirs(path)
        fs.make_dirs(path, exists_ok=True)
        self.assertRaises(FileExistsError, lambda: fs.make_dirs(path))

    def test_save_and_read_text_file_under_root(self):
        fs = FakeFileSystemBackend()
        f = fs.open_file("myfile", "w")
        f.write("hello")
        f.close()

        f = fs.open_file("myfile", "r")
        self.assertEqual("hello", f.read())
        f.close()

    def test_save_and_read_text_file_in_directory(self):
        fs = FakeFileSystemBackend()
        dir_path = os.path.join("my", "subdirectory")
        fs.make_dirs(dir_path, exists_ok=True)

        file_path = os.path.join(dir_path, 'myfile')
        f = fs.open_file(file_path, "w")
        f.write("hello")
        f.close()

        file_path = os.path.join(dir_path, 'myfile')
        f = fs.open_file(file_path, "r")
        self.assertEqual("hello", f.read())
        f.close()

    def test_save_and_read_binary_file(self):
        fs = FakeFileSystemBackend()
        dir_path = os.path.join("my", "subdirectory")
        fs.make_dirs(dir_path, exists_ok=True)

        data = b'hello'
        file_path = os.path.join(dir_path, 'myfile')
        f = fs.open_file(file_path, "wb")
        f.write(data)
        f.close()

        file_path = os.path.join(dir_path, 'myfile')
        f = fs.open_file(file_path, "rb")
        self.assertEqual(data, f.read())
        f.close()

    def test_cannot_operate_on_closed_file(self):
        fs = FakeFileSystemBackend()
        f = fs.open_file("some_file", "wb")
        f.close()
        self.assertRaises(ValueError, lambda: f.write(b'data'))

        f = fs.open_file("some_file", "rb")
        f.close()
        self.assertRaises(ValueError, f.read)

    def test_writing_to_file_open_for_reading_and_vice_versa(self):
        fs = FakeFileSystemBackend()
        f = fs.open_file("some_file", "w")
        self.assertRaises(UnsupportedOperation, f.read)
        f.close()
        f = fs.open_file("some_file", "rb")

        self.assertRaises(UnsupportedOperation, lambda: f.write(b'data'))
        f.close()


if __name__ == '__main__':
    unittest.main()
