import unittest
import json
from io import BytesIO
import hashlib

from distllm.control_center import ControlCenter, ModelSlice, NodeProvisioningError
from distllm.control_center import Connection, OperationFailedError
from distllm import protocol
from tests.unit.mocks import ComplexServerSocketMock


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

        self.assertEqual(1, len(self.socket.recorded_requests))
        self.assertEqual(protocol.RequestStatus(), self.socket.recorded_requests[0])

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

        self.assertEqual(1, len(self.socket.recorded_requests))
        self.assertEqual(protocol.RequestAllSlices(), self.socket.recorded_requests[0])

    def test_load_slice(self):
        expected = {
            'name': 'funky',
            'model': 'falcon'
        }

        self.socket.set_reply_body("load_slice_request", body=expected)
        self.assertEqual(expected, self.connection.load_slice('funky'))

        self.assertEqual(1, len(self.socket.recorded_requests))
        expected_request = protocol.RequestLoadSlice(name='funky')
        self.assertEqual(expected_request, self.socket.recorded_requests[0])

    def test_load_slice_unsuccessful(self):
        response_body = {
            'operation': 'load_slice',
            'error': 'Brief error',
            'description': 'Details'
        }
        self.socket.set_error_body("load_slice_request", body=response_body)
        self.assertRaises(OperationFailedError,
                          lambda: self.connection.load_slice('funky'))

    def test_propagate_forward_failure(self):
        response_body = {
            'operation': 'propagate_forward',
            'error': 'Out of memory',
            'description': 'Input tensor was too big'
        }
        self.socket.set_error_body("propagate_forward_request", body=response_body)

        self.assertRaises(OperationFailedError,
                          lambda: self.connection.propagate_forward([32, 32], (1, 2)))

    def test_propagate_forward_other_failure(self):
        response_body = {
            'axis0': 10,
            'axis1': 30,
            'values': list(range(200))
        }

        self.socket.set_reply_body("propagate_forward_request", body=response_body)

        shape = [10, 20]
        tensor_to_send = list(range(200))
        self.assertRaises(OperationFailedError,
                          lambda: self.connection.propagate_forward(tensor_to_send, shape))

    def test_propagate_forward_succeeds(self):
        response_body = {
            'axis0': 10,
            'axis1': 20,
            'values': list(range(200))
        }

        self.socket.set_reply_body("propagate_forward_request", body=response_body)

        input_tensor = list(range(200))
        shape = [10, 20]
        result = self.connection.propagate_forward(input_tensor, tuple(shape))

        self.assertEqual(1, len(self.socket.recorded_requests))
        expected_request = protocol.RequestPropagateForward(
            axis0=10, axis1=20, values=input_tensor
        )
        self.assertEqual(expected_request, self.socket.recorded_requests[0])

        self.assertEqual(shape, result['shape'])
        self.assertEqual(len(input_tensor), len(result['values']))
        for i in range(len(input_tensor)):
            expected_value = response_body['values'][i]
            self.assertAlmostEqual(expected_value, result['values'][i], places=4)

    def test_push_file_succeeds(self):
        data = b'Hello, World'

        begin_response_body = {
            'submission_id': 832
        }

        part_response_body = {
            'part_size': 4
        }

        end_response_body = {
            'file_name': 'funky',
            'total_size': len(data)
        }

        self.socket.set_reply_body("request_file_submission_begin", body=begin_response_body)
        self.socket.set_reply_body("request_submit_part", body=part_response_body)
        self.socket.set_reply_body("request_file_submission_end", body=end_response_body)

        f = BytesIO(data)
        metadata = {"field": "value", "another": 32}
        metadata_json = json.dumps(metadata)
        result = self.connection.push_file(f, metadata, chunk_size=4)

        self.assertEqual(end_response_body, result)

        # assert that server received requests and data as expected

        self.assertEqual(5, len(self.socket.recorded_requests))

        begin_request = self.socket.recorded_requests[0]
        self.assertEqual(protocol.RequestFileSubmissionBegin(metadata_json), begin_request)

        # assert server received valid submission id, part number and data
        # request sending part # 1
        part_request1 = self.socket.recorded_requests[1]
        expected_request = protocol.RequestSubmitPart(
            submission_id=832, part_number=0, data=data[:4]
        )
        self.assertEqual(expected_request, part_request1)
        
        # request sending part # 2
        part_request2 = self.socket.recorded_requests[2]
        expected_request = protocol.RequestSubmitPart(
            submission_id=832, part_number=1, data=data[4:8]
        )
        self.assertEqual(expected_request, part_request2)

        # request sending part # 3
        part_request3 = self.socket.recorded_requests[3]
        expected_request = protocol.RequestSubmitPart(
            submission_id=832, part_number=2, data=data[8:]
        )
        self.assertEqual(expected_request, part_request3)

        # check that server receives submission_id and checksum
        submission_end_request = self.socket.recorded_requests[4]
        checksum = hashlib.sha256(data).hexdigest()
        expected_request = protocol.RequestFileSubmissionEnd(submission_id=832,
                                                             checksum=checksum)
        self.assertEqual(expected_request, submission_end_request)

    def test_push_file_succeeds_with_chunk_resubmission(self):
        data = b'Hello, World'

        begin_response_body = {
            'submission_id': 832
        }

        count = 0

        def part_response():
            nonlocal count
            count += 1
            if count < 3:
                return protocol.ResponseWithError('request_submit_part', 'integrity_error', '')

            return protocol.ResponseSubmitPart(4)

        end_response_body = {
            'file_name': 'funky',
            'total_size': len(data)
        }

        self.socket.set_reply_body("request_file_submission_begin", body=begin_response_body)
        self.socket.set_reply_function("request_submit_part", part_response)
        self.socket.set_reply_body("request_file_submission_end", body=end_response_body)

        f = BytesIO(data)
        result = self.connection.push_file(f, chunk_size=4)

        self.assertEqual(end_response_body, result)

    def test_push_file_fails_on_submission_begin(self):
        data = b'Hello, World'

        response_body = {
            'operation': 'request_file_submission_begin',
            'error': 'Out of memory',
            'description': ''
        }

        self.socket.set_error_body("request_file_submission_begin", body=response_body)

        f = BytesIO(data)

        self.assertRaises(OperationFailedError,
                          lambda: self.connection.push_file(f, chunk_size=4))
 
    def test_push_file_fails_on_submitting_part(self):
        data = b'Hello, World'

        begin_response_body = {
            'submission_id': 832
        }

        part_response_body = {
            'operation': 'request_submit_part',
            'error': 'Out of memory',
            'description': ''
        }

        self.socket.set_reply_body("request_file_submission_begin", body=begin_response_body)
        self.socket.set_error_body("request_submit_part", body=part_response_body)

        f = BytesIO(data)

        self.assertRaises(OperationFailedError,
                    lambda: self.connection.push_file(f, chunk_size=4))
 
    def test_push_file_fails_on_finalization(self):
        data = b'Hello, World'

        begin_response_body = {
            'submission_id': 832
        }

        part_response_body = {
            'part_size': 4
        }

        end_response_body = {
            'operation': 'request_file_submission_end',
            'error': 'Out of memory',
            'description': ''
        }

        self.socket.set_reply_body("request_file_submission_begin", body=begin_response_body)
        self.socket.set_reply_body("request_submit_part", body=part_response_body)
        self.socket.set_error_body("request_file_submission_end", body=end_response_body)

        f = BytesIO(data)
        self.assertRaises(OperationFailedError,
                          lambda: self.connection.push_file(f, chunk_size=4))
 

if __name__ == '__main__':
    unittest.main()
