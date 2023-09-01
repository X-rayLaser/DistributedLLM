import unittest
import os
import json
import hashlib
from io import BytesIO
from distllm.compute_node import (
    TCPHandler, RequestContext, FailingSliceContainer
)
from distllm.compute_node import routes
from distllm.compute_node import (
    FailedUploadError, UploadNotFoundError, ParallelUploadError, NoActiveUploadError
)
from distllm.compute_node.uploads import (
    FileUpload, UploadManager, UploadRegistry, FunkyNameGenerator
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
        context = RequestContext.default(names=names)
        request_handler = TCPHandler(socket, context)
        
        upload_slices(context.manager)

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
        context = RequestContext.default(names=names)
        request_handler = TCPHandler(socket, context)
        
        upload_slices(context.manager)

        model = 'falcon'
        load_slice = 'second slice'
        request = protocol.RequestLoadSlice(name=load_slice)
        request_data = request.encode()
        socket.inject_data(request_data)
        request_handler.handle()

        msg, body = protocol.receive_message(socket)
        message = protocol.restore_message(msg, body)
        self.assertEqual(protocol.JsonResponseWithLoadedSlice(load_slice, model), message)

    def test_request_load_non_existing_slice(self):
        socket = mocks.StableSocketMock()

        names = ['first slice', 'second slice']
        context = RequestContext.default(names=names)
        request_handler = TCPHandler(socket, context)
        
        upload_slices(context.manager)

        load_slice = 'missing slice'
        request = protocol.RequestLoadSlice(name=load_slice)
        request_data = request.encode()
        socket.inject_data(request_data)
        request_handler.handle()

        msg, body = protocol.receive_message(socket)
        message = protocol.restore_message(msg, body)
        self.assertEqual(protocol.ResponseWithError(request.msg, "slice_not_found", ""), message)

    def test_load_slice_results_in_error(self):
        names = ["first", "second"]
        context = RequestContext.with_failing_loader(names=names)
        upload_slices(context.manager)
        handler = routes.LoadSliceHandler(context)
        
        request_message = protocol.RequestLoadSlice("first")
        response_message = handler(request_message)
        expected = protocol.ResponseWithError(
            request_message.get_message(), "slice_load_error", ""
        )
        self.assertEqual(expected, response_message)

    def test_begin_file_upload_fails(self):
        context = RequestContext.default()
        metadata = dict(type='any_file')
        context.manager.prepare_upload(metadata)

        handler = routes.FileSubmissionBeginHandler(context)
        
        request_message = protocol.RequestFileSubmissionBegin(json.dumps(metadata))
        response_message = handler(request_message)

        expected = protocol.ResponseWithError(request_message.get_message(),
                                              "parallel_upload_forbidden", "")
        self.assertEqual(expected, response_message)

    def test_begin_file_upload_succeeds(self):
        context = RequestContext.default(names=["first", "second"])
        upload_slices(context.manager)

        handler = routes.FileSubmissionBeginHandler(context)
        
        metadata = dict(type='any_file')
        request_message = protocol.RequestFileSubmissionBegin(json.dumps(metadata))
        response_message = handler(request_message)

        expected = protocol.ResponseFileSubmissionBegin(3)
        self.assertEqual(expected, response_message)

    def test_upload_file_part_fails_upload_not_found(self):
        context = RequestContext.default(names=["first", "second"])
        upload_slices(context.manager)

        handler = routes.SubmitPartHandler(context)
        
        request_message = protocol.RequestSubmitPart(
            submission_id=32, part_number=0, data=b'data'
        )

        response_message = handler(request_message)

        expected = protocol.ResponseWithError(request_message.get_message(),
                                              error='upload_not_found',
                                              description='')

        self.assertEqual(expected, response_message)

    def test_upload_file_part_succeeds(self):
        context = RequestContext.default(names=["first", "second"])
        metadata = dict(type='any_file')
        submit_id = context.manager.prepare_upload(metadata)

        handler = routes.SubmitPartHandler(context)
        
        data = b'data'
        request_message = protocol.RequestSubmitPart(
            submission_id=submit_id, part_number=0, data=data
        )

        response_message = handler(request_message)

        expected = protocol.ResponseSubmitPart(len(data))

        self.assertEqual(expected, response_message)


class SubmissionEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = RequestContext.default(names=["first", "second"])
        
        metadata = dict(type='any_file')
        self.submit_id = self.context.manager.prepare_upload(metadata)
        self.data = b'data'
        self.context.manager.upload_part(self.submit_id, self.data)

        self.handler = routes.FileSubmissionEndHandler(self.context)

    def test_submission_end_fails(self):
        invalid_checksum = '8239823abc823'
        request_message = protocol.RequestFileSubmissionEnd(self.submit_id, invalid_checksum)

        response_message = self.handler(request_message)

        expected = protocol.ResponseWithError(request_message.get_message(),
                                              error="file_upload_failed",
                                              description="")

        self.assertEqual(expected, response_message)

    def test_submission_end_failes_because_of_wrong_id(self):
        request_message = protocol.RequestFileSubmissionEnd(834, checksum='')
        response_message = self.handler(request_message)

        expected = protocol.ResponseWithError(request_message.get_message(),
                                              error="upload_not_found",
                                              description="")

        self.assertEqual(expected, response_message)

    def test_submission_end_fails_because_name_generator_exhausted(self):
        self.context.name_gen.names = []

        checksum = hashlib.sha256(self.data).hexdigest()
        request_message = protocol.RequestFileSubmissionEnd(self.submit_id, checksum)

        handler = routes.FileSubmissionEndHandler(self.context)
        response_message = handler(request_message)

        expected = protocol.ResponseWithError(request_message.get_message(),
                                              error="file_upload_failed",
                                              description="")
        self.assertEqual(expected, response_message)

    def test_submission_end_succeeds(self):
        checksum = hashlib.sha256(self.data).hexdigest()
        request_message = protocol.RequestFileSubmissionEnd(self.submit_id, checksum)

        response_message = self.handler(request_message)

        expected = protocol.ResponseFileSubmissionEnd(file_name="first",
                                                      total_size=len(self.data))
        self.assertEqual(expected, response_message)


class PropagateForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = RequestContext.default(names=["first", "second", "third"])
        self.k, self.b = self._push_dummy_slice(self.context)


        self.send_values = [1, 2, 3, 4, 5, 6]
        self.request_message = protocol.RequestPropagateForward(
            axis0=2, axis1=3, values=self.send_values
        )

    def test_cannot_compute_when_no_slice_loaded(self):
        handler = routes.PropagateForwardHandler(self.context)
        expected = protocol.ResponseWithError(operation=self.request_message.get_message(),
                                              error="slice_not_loaded",
                                              description="")

        response_message = handler(self.request_message)
        self.assertEqual(expected, response_message)

    def test_propagate_forward_results_in_error(self):
        fs_backend = FakeFileSystemBackend()
        self.context.slice_container = FailingSliceContainer(fs_backend)

        handler = routes.PropagateForwardHandler(self.context)
        expected = protocol.ResponseWithError(operation=self.request_message.get_message(),
                                              error="neural_computation_error",
                                              description="")

        response_message = handler(self.request_message)
        self.assertEqual(expected, response_message)

    def test_compute_tensor(self):
        message = protocol.RequestLoadSlice("first")
        handler = routes.LoadSliceHandler(self.context)
        res = handler(message)
        print(res.get_message(), res.get_body())

        handler = routes.PropagateForwardHandler(self.context)
        response_message = handler(self.request_message)

        expected_values = [self.k * v + self.b for v in self.send_values]
        expected = protocol.ResponsePropagateForward(
            axis0=2, axis1=3, values=expected_values
        )

        self.assertEqual(expected, response_message)

    def _push_dummy_slice(self, context):
        k = 10
        b = 150
        model_data = bytes([k, b])  # computes kx + b elementwise
        metadata = dict(type='slice', format='test', model='testmodel', layer_from=0, layer_to=0)

        submit_id = context.manager.prepare_upload(metadata)
        context.manager.upload_part(submit_id, model_data)

        checksum = hashlib.sha256(model_data).hexdigest()
        context.manager.finilize_upload(submit_id, checksum)
        return k, b


# todo: get_status tests


def upload_slices(manager):
    metadata = dict(type='slice', format='test', model='llama_v1', layer_from=0, layer_to=12)
    generate_fake_data(manager, metadata)
    metadata = dict(type='slice', format='test', model='falcon', layer_from=12, layer_to=28)
    generate_fake_data(manager, metadata)
    metadata = dict(type='any_file')
    generate_fake_data(manager, metadata)


def generate_fake_data(manager, metadata):
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
