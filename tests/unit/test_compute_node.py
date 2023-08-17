import unittest
import os
import json
import hashlib
from io import BytesIO
from distllm.compute_node import (
    FileUpload, UploadRegistry, UploadManager,
    TCPHandler
)
from distllm.compute_node import (
    FailedUploadError, UploadNotFoundError, ParallelUploadError, NoActiveUploadError
)
from tests.unit import mocks
from distllm import protocol
from distllm.utils import FakeFileSystemBackend, receive_data


class ServerResponseTests(unittest.TestCase):
    def test_list_existing_slices(self):
        expected_slices = [{
            'name': 'first slice',
            'model': 'llama_v1',
            'layer_from': 0,
            'layer_to': 12
        }, {
            'name': 'second slice',
            'model': 'falcon',
            'layer_from': 12,
            'layer_to': 28
        }]
        slices_json = json.dumps(expected_slices)

        socket = mocks.StableSocketMock()

        names = ['first slice', 'second slice']
        request_handler = TCPHandler(socket, names)
        
        self._upload_slice(request_handler.manager)

        request = protocol.RequestAllSlices()
        request_data = request.encode()
        socket.inject_data(request_data)
        request_handler.handle()
        
        msg, body = protocol.receive_message(socket)
        message = protocol.restore_message(msg, body)
        self.assertEqual(protocol.JsonResponseWithSlices(slices_json), message)

    def test_request_load_slice(self):
        socket = mocks.StableSocketMock()

        names = ['first slice', 'second slice']
        request_handler = TCPHandler(socket, names)
        
        self._upload_slice(request_handler.manager)

        model = 'falcon'
        load_slice = 'second slice'
        request = protocol.RequestLoadSlice(name=load_slice)
        request_data = request.encode()
        socket.inject_data(request_data)
        request_handler.handle()

        msg, body = protocol.receive_message(socket)
        message = protocol.restore_message(msg, body)
        self.assertEqual(protocol.JsonResponseWithLoadedSlice(load_slice, model), message)

    def _upload_slice(self, manager):
        metadata = dict(type='slice', model='llama_v1', layer_from=0, layer_to=12)
        self._generate_fake_data(manager, metadata)
        metadata = dict(type='slice', model='falcon', layer_from=12, layer_to=28)
        self._generate_fake_data(manager, metadata)
        metadata = dict(type='any_file')
        self._generate_fake_data(manager, metadata)

    def _generate_fake_data(self, manager, metadata):
        submit_id = manager.prepare_upload(metadata)
        manager.upload_part(submit_id, b'data')
        manager.finilize_upload(submit_id, hashlib.sha256(b'data').hexdigest())


class UploadManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = UploadRegistry('uploads')
        self.manager = UploadManager(self.registry)
        self.manager.fs_backend = FakeFileSystemBackend()

    def test_successful_upload(self):
        metadata = dict(type='slice', model='mymodel')
        submit_id = self.manager.prepare_upload(metadata)

        data = bytes([82, 102, 255, 0, 123])
        checksum = hashlib.sha256(data).hexdigest()
        num_bytes = self.manager.upload_part(submit_id, data[:3])
        self.assertEqual(3, num_bytes)
        num_bytes = self.manager.upload_part(submit_id, data[3:])
        self.assertEqual(2, num_bytes)
        
        total_size = self.manager.finilize_upload(submit_id, checksum)
        self.assertEqual(len(data), total_size)

        self.assertEqual([0], self.registry.finished)

    def test_all_possible_failures(self):
        slice_meta = dict(type='slice', model='mymodel')
        afile_meta = dict(type='any_file')
        submit_id = self.manager.prepare_upload(slice_meta)
        self.assertRaises(ParallelUploadError, lambda: self.manager.prepare_upload(afile_meta))
        self.assertRaises(UploadNotFoundError, lambda: self.manager.upload_part(83, b''))

        self.manager.upload_part(submit_id, b'abcd')
        self.manager.upload_part(submit_id, b'efg')
        self.assertRaises(
            FailedUploadError,
            lambda: self.manager.finilize_upload(submit_id, hashlib.sha256(b'abc').hexdigest)
        )
        self.assertEqual([], self.registry.finished)
        self.assertEqual([0], self.registry.failed)

    def test_make_few_uploads(self):
        slice_meta = dict(type='slice', model='mymodel')
        afile_meta = dict(type='any_file')
        id1 = self.manager.prepare_upload(slice_meta)

        blob1 = b'12345'
        blob2 = b'6789'

        self.manager.upload_part(id1, blob1)
        self.manager.upload_part(id1, blob2)

        upload_size1 = self.manager.finilize_upload(id1, hashlib.sha256(blob1 + blob2).hexdigest())
        self.assertEqual(len(blob1 + blob2), upload_size1)
    
        id2 = self.manager.prepare_upload(afile_meta)

        blob3 = bytes([93, 88, 0, 123, 254])
        self.manager.upload_part(id2, blob3)

        self.assertRaises(
            FailedUploadError,
            lambda: self.manager.finilize_upload(id2, hashlib.sha256().hexdigest)
        )

        id3 = self.manager.prepare_upload(afile_meta)

        self.manager.upload_part(id3, blob3)

        upload_size2 = self.manager.finilize_upload(id3, hashlib.sha256(blob3).hexdigest())
        self.assertEqual(len(blob3), upload_size2)
        self.assertEqual([id2], self.registry.failed)
        self.assertEqual([id1, id3], self.registry.finished)


class SingleUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        ram_buffer = BytesIO()
        self.upload = FileUpload(ram_buffer)

    def test_empty_upload(self):
        self.assertEqual(0, self.upload.total_size())
        self.assertEqual(hashlib.sha256().hexdigest(), self.upload.checksum())

    def test_upload_single_byte(self):
        self.upload.update(b'h')
        self.assertEqual(1, self.upload.total_size())
        self.assertEqual(hashlib.sha256(b'h').hexdigest(), self.upload.checksum())

    def test_upload_multiple_bytes_at_once(self):
        s = b'hello, world!'
        self.upload.update(s)
        self.assertEqual(len(s), self.upload.total_size())
        self.assertEqual(hashlib.sha256(s).hexdigest(), self.upload.checksum())

    def test_upload_one_byte_at_a_time(self):
        s = b'hello, world!'
        for b in s:
            self.upload.update(bytes([b]))

        self.assertEqual(len(s), self.upload.total_size())
        self.assertEqual(hashlib.sha256(s).hexdigest(), self.upload.checksum())

    def test_upload_by_chunks(self):
        s = b'hello, world!'
        self.upload.update(b'hello,')
        self.upload.update(b' world')
        self.upload.update(b'!')

        self.assertEqual(len(s), self.upload.total_size())
        self.assertEqual(hashlib.sha256(s).hexdigest(), self.upload.checksum())

    def test_valid_upload(self):
        s = b'hello, world!'
        self.upload.update(b'hello,')
        self.upload.update(b' world')
        self.upload.update(b'!')
        self.upload.validate(hashlib.sha256(s).hexdigest())

    def test_failed_upload(self):
        s = b'hello, world!'
        checksum = hashlib.sha256(s).hexdigest()
        self.upload.update(b'hello,')
        self.upload.update(b' world')
        self.upload.update(b'#')
        self.assertRaises(FailedUploadError, lambda: self.upload.validate(checksum))

    def test_uploading_long_byte_sequence(self):
        s = bytes([123] * 10000)
        checksum = hashlib.sha256(s).hexdigest()
        self.upload.update(s[:2000])
        self.upload.update(s[2000:])
        self.upload.validate(checksum)


class UploadRegistryTests(unittest.TestCase):
    def setUp(self):
        self.root = 'uploads'
        self.registry = UploadRegistry(self.root)

    def test_initial_registry(self):
        self.assertEqual([], self.registry.in_progress)
        self.assertEqual([], self.registry.failed)
        self.assertEqual([], self.registry.finished)
        
        self.assertRaises(UploadNotFoundError, lambda: self.registry.get_location(34))
        self.assertRaises(NoActiveUploadError, lambda: self.registry.mark_finished())
        self.assertRaises(NoActiveUploadError, lambda: self.registry.mark_failed())

    def test_add_one_upload(self):
        metadata = dict(type='slice', model='llama')

        submit_id = self.registry.add_upload(metadata)
        self.assertEqual(0, submit_id)
        self.assertEqual([submit_id], self.registry.in_progress)

        self.assertEqual([], self.registry.failed)
        self.assertEqual([], self.registry.finished)

    def test_location_of_upload_for_slices(self):
        metadata = dict(type='slice', model='llama')

        submit_id = self.registry.add_upload(metadata)

        expected_base = os.path.join(self.root, self.registry.slices_dir, "upload_0")
        expected_upload_path = os.path.join(expected_base, self.registry.uploaded_file)

        upload_location = self.registry.get_location(submit_id)
        self.assertEqual(expected_upload_path, upload_location.upload_path)

        expected_metadata_path = os.path.join(expected_base, self.registry.metadata_file)
        self.assertEqual(expected_metadata_path, upload_location.metadata_path)

    def test_location_of_upload_for_other_files(self):
        metadata = dict(type='any_file')

        submit_id = self.registry.add_upload(metadata)

        expected_base = os.path.join(self.root, self.registry.other_dir, "upload_0")
        expected_upload_path = os.path.join(expected_base, self.registry.uploaded_file)

        expected_metadata_path = os.path.join(expected_base, self.registry.metadata_file)

        upload_location = self.registry.get_location(submit_id)
        self.assertEqual(expected_upload_path, upload_location.upload_path)
        self.assertEqual(expected_metadata_path, upload_location.metadata_path)

    def test_no_concurrent_uploads_allowed(self):
        metadata = dict(type='slice', model='llama')

        self.registry.add_upload(metadata)

        self.assertRaises(ParallelUploadError, lambda: self.registry.add_upload(metadata))

    def test_upload_files_sequentially(self):
        metadata = dict(type='any_file', model='llama')

        id1 = self.registry.add_upload(metadata)
        self.registry.mark_finished()

        id2 = self.registry.add_upload(metadata)
        self.registry.mark_failed()

        id3 = self.registry.add_upload(metadata)
        self.registry.mark_finished()

        self.assertEqual([id1, id3], self.registry.finished)
        self.assertEqual([id2], self.registry.failed)

    def test_persistence(self):
        metadata = dict(type='any_file')

        id1 = self.registry.add_upload(metadata)
        self.registry.mark_finished()

        id2 = self.registry.add_upload(metadata)
        self.registry.mark_failed()

        metadata = dict(type='slice', model='llama')
        id3 = self.registry.add_upload(metadata)
        self.registry.mark_finished()

        state_json = self.registry.to_json()

        copied_registry = UploadRegistry.from_json(state_json)

        self.assertEqual(state_json, copied_registry.to_json())

        self.assertEqual([id1, id3], self.registry.finished)
        self.assertEqual([id2], self.registry.failed)
