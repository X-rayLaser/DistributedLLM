import unittest
import os
import json
import hashlib
from io import BytesIO
from distllm.compute_node import (
    FileUpload, UploadRegistry, UploadManager,
    DebugFileSystemBackend, FakeFileSystemBackend, TCPHandler, FakeFileTree
)
from distllm.compute_node import (
    FailedUploadError, UploadNotFoundError, ParallelUploadError, NoActiveUploadError
)
from tests.unit import mocks
from distllm import protocol
from distllm.utils import receive_data


class ServerResponseTests(unittest.TestCase):
    def test_list_slices(self):
        return
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

        request_handler = TCPHandler(socket)
        request_handler.manager.fs_backend = DebugFileSystemBackend()

        request = protocol.RequestAllSlices()
        request_data = request.encode()
        socket.inject_data(request_data)
        request_handler.handle()
        
        msg, body = protocol.receive_message(socket)
        message = protocol.restore_message(msg, body)
        self.assertEqual(protocol.JsonResponseWithSlices(slices_json), message)


class UploadManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = UploadRegistry('uploads')
        self.manager = UploadManager(self.registry)
        self.manager.fs_backend = DebugFileSystemBackend()

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


class FakeFileTreeTests(unittest.TestCase):
    def test_empty_tree(self):
        tree = FakeFileTree()
        self.assertFalse(tree.exists('foo'))
        self.assertEqual([], tree.list_files())
        self.assertEqual([], tree.list_dirs())

    def test_adding_a_file_into_missing_folder(self):
        tree = FakeFileTree()
        self.assertRaises(FileNotFoundError,
                          lambda: tree.add_file(os.path.join("foo", "bar.txt")))

    def test_putting_a_file_under_the_root(self):
        tree = FakeFileTree('')

        self.assertRaises(Exception, lambda: tree.add_file(''))

        tree.add_file("file_name")
        self.assertEqual(["file_name"], tree.list_files())
        self.assertEqual([], tree.list_dirs())

    def test_adding_folder_under_root(self):
        tree = FakeFileTree('')

        self.assertRaises(Exception, lambda: tree.make_dirs(''))

        tree.make_dirs(os.path.join("foo"))
        self.assertEqual(["foo"], tree.list_dirs())

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
        self.assertRaises(FileExistsError,
                          lambda: tree.make_dirs(os.path.join('second_file', 'subdir')))

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

    def test_existence_of_folders(self):
        tree = FakeFileTree('')

        tree.make_dirs(os.path.join("foo", "bar"))

        self.assertTrue(tree.exists("foo"))
        self.assertTrue(tree.exists(os.path.join("foo", "bar")))

        self.assertFalse(tree.exists("bar"))


class FakeFileSystemTests(unittest.TestCase):
    def test_exceptions(self):
        fs = FakeFileSystemBackend()
        self.assertRaises(FileNotFoundError, lambda: fs.open_file("some_file", "r"))
        self.assertRaises(FileNotFoundError,
                          lambda: fs.open_file(os.path.join("some_dir", "somefile"), "w"))

        path = os.path.join("some_dir", "subdir")
        fs.make_dirs(path)
        self.assertEqual(FileExistsError, lambda: fs.make_dirs(path))

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
