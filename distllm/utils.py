import io
import os
from pathlib import Path
import struct


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

    def parse_float(self):
        return self._parse(self.coder.decode_float)

    def parse_list(self):
        return self._parse(self.coder.decode_list)

    def _parse(self, decode_func):
        value, new_pos = decode_func(self.stream[self.pos:])
        self.pos += new_pos
        return value


class ByteCoder:
    def encode_int(self, value):
        return self._encode_number('i', value)

    def decode_int(self, value_bytes):
        if len(value_bytes) < 4:
            raise ByteCodingError
        
        new_byte_position = 4
        return self._decode_number('i', value_bytes[:4]), new_byte_position

    def encode_float(self, value):
        return self._encode_number('f', value)

    def decode_float(self, float_bytes):
        new_pos = 4
        return self._decode_number('f', float_bytes[:new_pos]), new_pos

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

    def encode_list(self, alist):
        size = len(alist)
        size_bytes = self.encode_int(size)

        barr = bytearray()
        barr.extend(size_bytes)
        for element in alist:
            barr.extend(self.encode_float(element))
        
        return bytes(barr)

    def decode_list(self, list_bytes):
        value_size = 4
        num_elems, list_offset = self.decode_int(list_bytes[:value_size])
        if len(list_bytes) - list_offset < num_elems * value_size:
            raise ByteCodingError
        
        res = []
        for i in range(num_elems):
            pos = i * value_size + list_offset
            value = struct.unpack('f', list_bytes[pos:pos + value_size])[0]
            res.append(value)
        return res, list_offset + num_elems * value_size

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

    def _encode_number(self, format, value):
        try:
            return struct.pack(format, value)
        except struct.error as e:
            raise ByteCodingError(str(e))

    def _decode_number(self, format, value_bytes):
        try:
            return struct.unpack(format, value_bytes)[0]
        except struct.error as e:
            raise ByteCodingError(str(e))


class ByteCodingError(Exception):
    pass


def encode_length(length):
    return struct.pack('i', length)


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


class FakeFileTree:
    def __init__(self, root='/'):
        self.root = root

        self.tree = {}
        self.root_dir = {
            'file': False,
            'tree': self.tree
        }

    def make_dirs(self, path):
        if self.exists(path):
            raise FileExistsError

        parts = Path(path).parts

        if not parts:
            raise Exception('Empty path')

        def add(tree, rem_dirs):
            if not rem_dirs:
                return

            name, *rem_dirs = rem_dirs

            if name not in tree:
                sub_tree = {}
                tree[name] = {'file': False, 'tree': sub_tree}
            else:
                d = tree[name]
                if d['file']:
                    raise FileExistsError(name)

                sub_tree = d['tree']
            add(sub_tree, rem_dirs)

        add(self.tree, parts)

    def add_file(self, path):
        *dirs, file_name = Path(path).parts

        dir_path = os.path.join(*dirs) if dirs else ''
        if dir_path:
            dir_obj = self._find_dir(dir_path)
            sub_tree = dir_obj['tree']
        else:
            sub_tree = self.tree

        if file_name in sub_tree:
            raise FileExistsError

        sub_tree[file_name] = {'file': True, 'data': b''}

    def _get_object(self, path):
        parts = Path(path).parts
        if not parts:
            raise FileNotFoundError
        *dirs, file_name = parts
        sub_tree = self.tree
        obj = None
        for part in dirs:
            obj = sub_tree.get(part)
            if obj is None or obj['file']:
                raise FileNotFoundError

            sub_tree = obj['tree']

        try:
            obj = sub_tree[file_name]
        except KeyError:
            raise FileNotFoundError

        return obj

    def _find_dir(self, path):
        obj = self._get_object(path)
        if obj['file']:
            raise FileNotFoundError

        return obj

    def find_file(self, path):
        obj = self._get_object(path)
        if not obj['file']:
            raise FileNotFoundError

        return obj

    def write_to_file(self, path, data):
        obj = self.find_file(path)
        obj['data'] = data

    def read_file(self, path):
        obj = self.find_file(path)
        return obj['data']

    def exists(self, path):
        path = os.path.join(self.root, path)
        return path in self.list_files() or path in self.list_dirs()

    def list_files(self):
        files, _ = self._list_objects()
        return files

    def list_dirs(self):
        _, dirs = self._list_objects()
        return dirs

    def _list_objects(self):
        files = []
        dirs = []
        def traverse(name, d, path):
            current_path = os.path.join(path, name)
            if d['file']:
                files.append(current_path)
            else:
                dirs.append(current_path)

                for k, container in d['tree'].items():
                    traverse(k, container, current_path)

        traverse(self.root, self.root_dir, '')

        if self.root in files:
            files.remove(self.root)

        if self.root in dirs:
            dirs.remove(self.root)
        return files, dirs


class FileSystemBackend:
    def make_dirs(self, path, exists_ok=False):
        raise NotImplementedError

    def open_file(self, path, mode):
        raise NotImplementedError


class DefaultFileSystemBackend(FileSystemBackend):
    def make_dirs(self, path, exists_ok=False):
        os.makedirs(path, exist_ok=exists_ok)

    def open_file(self, path, mode):
        return open(path, mode)


class DebugFileSystemBackend(FileSystemBackend):
    def make_dirs(self, path, exists_ok=False):
        pass

    def open_file(self, path, mode):
        if 'b' in mode:
            return io.BytesIO()
        else:
            return io.StringIO()


class FakeFileSystemBackend(FileSystemBackend):
    class FakeFile:
        def __init__(self, file_obj, mode):
            self.file_obj = file_obj
            self.mode = mode
            self.closed = False

            if mode == 'w' or mode == 'wb':
                self.file_obj['data'] = b''

        def read(self, max_chunk=None):
            if self.closed:
                raise ValueError('I/O operation on closed file')

            if not self.is_readable():
                raise io.UnsupportedOperation('not readable')

            data = self.file_obj['data']
            if 'b' in self.mode:
                return data
            return data.decode('utf-8')

        def write(self, data, max_chunk=None):
            if self.closed:
                raise ValueError('I/O operation on closed file')

            if not self.is_writable():
                raise io.UnsupportedOperation('not writable')

            if 'b' not in self.mode:
                data = data.encode('utf-8')
            self.file_obj['data'] += data

        def close(self):
            self.closed = True

        def is_readable(self):
            return 'r' in self.mode

        def is_writable(self):
            return self.mode == 'r+' or 'w' in self.mode

    def __init__(self):
        self.file_tree = FakeFileTree()

    def make_dirs(self, path, exists_ok=False):
        try:
            self.file_tree.make_dirs(path)
        except FileExistsError:
            if not exists_ok:
                raise

    def open_file(self, path, mode):
        if 'w' in mode:
            if not self.file_tree.exists(path):
                self.file_tree.add_file(path)

        file_object = self.file_tree.find_file(path)
        return self.FakeFile(file_object, mode)