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