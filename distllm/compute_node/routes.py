import json
from distllm import protocol
from distllm.compute_node.slices import Tensor, SliceNotLoadedError, NeuralComputationError
from .uploads import FailedUploadError, ParallelUploadError, UploadNotFoundError
routes = {}


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        routes[cls.request_name] = cls
        return cls


class RequestHandler(metaclass=Meta):
    request_name = ""

    def __init__(self, context):
        self.context = context

    def __call__(self, message):
        """Process incoming message and produce reply message"""
        raise NotImplemented


# todo: unittest this class
class RequestStatusHandler(RequestHandler):
    request_name = "status_request"

    def __call__(self, message):
        container = self.context.slice_container
        status_dict = {
            'status': 'up' if container.is_loaded else 'brand_new'
        }

        if container.is_loaded:
            status_dict['metadata'] = container.metadata
        status_json = json.dumps(status_dict)
        return protocol.JsonResponseWithStatus(status_json)


class GetAllSlicesHandler(RequestHandler):
    request_name = "slices_request"

    def __call__(self, message):
        slices = self.get_slices()
        slices_json = json.dumps(slices)
        return protocol.JsonResponseWithSlices(slices_json)

    def get_slices(self):
        ids = self.context.registry.finished

        slices = []
        for submission_id in ids:
            location = self.context.registry.get_location(submission_id)
            path = location.metadata_path
            f = self.context.manager.fs_backend.open_file(path, 'r')
            s = f.read()
            f.close()
            metadata = json.loads(s)
            if metadata['type'] != 'slice':
                continue

            model = metadata['model']
            layer_from = metadata['layer_from']
            layer_to = metadata['layer_to']
            #family = metadata.get('family', 'unknown')
            
            slice_name = self.context.name_gen.id_to_name(submission_id)
            slices.append(dict(name=slice_name, model=model,
                                layer_from=layer_from, layer_to=layer_to))
        return slices


class LoadSliceHandler(GetAllSlicesHandler):
    request_name = "load_slice_request"

    def __call__(self, message):
        slice_name = message.name
        
        # todo: simulate operation to load slice
        slices = self.get_slices()
        model = None
        for sl in slices:
            if sl['name'] == slice_name:
                model = sl['model']
                file_path = self._locate_file(slice_name)
                metadata = self._get_metadata(slice_name)

                try:
                    self.context.slice_container.load(file_path, metadata)
                except Exception:
                    return protocol.ResponseWithError(operation=message.msg,
                                                      error='slice_load_error',
                                                      description='')
 
                response = protocol.JsonResponseWithLoadedSlice(
                    name=slice_name, model=model
                )
                return response

        # if we got here, that means a given slice wasn't found
        return protocol.ResponseWithError(operation=message.msg, error='slice_not_found', description='')

    def _locate_file(self, slice_name):
        submission_id = self.context.name_gen.name_to_id(slice_name)
        location = self.context.registry.get_location(submission_id)
        path = location.upload_path
        return path
        return self.context.manager.fs_backend.open_file(path, mode='rb')

    def _get_metadata(self, slice_name):
        location = self._locate_upload(slice_name)
        f = self.context.manager.fs_backend.open_file(location.metadata_path, mode='r')
        metadata_json = f.read()
        f.close()
        return json.loads(metadata_json)

    def _locate_upload(self, slice_name):
        submission_id = self.context.name_gen.name_to_id(slice_name)
        return self.context.registry.get_location(submission_id)


class FileSubmissionBeginHandler(RequestHandler):
    request_name = "request_file_submission_begin"

    def __call__(self, message):
        metadata = json.loads(message.metadata_json)
        try:
            submission_id = self.context.manager.prepare_upload(metadata)
            return protocol.ResponseFileSubmissionBegin(submission_id)
        except ParallelUploadError:
            return protocol.ResponseWithError(operation=message.get_message(),
                                              error="parallel_upload_forbidden",
                                              description="")


class SubmitPartHandler(RequestHandler):
    request_name = "request_submit_part"

    def __call__(self, message):
        try:
            num_bytes = self.context.manager.upload_part(message.submission_id, message.data)
            return protocol.ResponseSubmitPart(num_bytes)
        except UploadNotFoundError:
            return protocol.ResponseWithError(operation=message.get_message(),
                                              error="upload_not_found",
                                              description="")


class FileSubmissionEndHandler(RequestHandler):
    request_name = "request_file_submission_end"

    def __call__(self, message):
        try:
            total_size = self.context.manager.finilize_upload(
                message.submission_id, message.checksum
            )
        except FileNotFoundError:
            return protocol.ResponseWithError(operation=message.get_message(),
                                              error="upload_not_found",
                                              description="")
        except FailedUploadError:
            return protocol.ResponseWithError(operation=message.get_message(),
                                              error="file_upload_failed",
                                              description="")
        else:
            name = self.context.name_gen.id_to_name(message.submission_id)
            if not name:
                return protocol.ResponseWithError(operation=message.get_message(),
                                              error="file_upload_failed",
                                              description="")
            return protocol.ResponseFileSubmissionEnd(name, total_size)


class PropagateForwardHandler(RequestHandler):
    request_name = "propagate_forward_request"
    
    def __call__(self, message):
        shape = (message.axis0, message.axis1)
        tensor = Tensor(shape, message.values)

        try:
            res_tensor = self.context.slice_container.forward(tensor)
        except NeuralComputationError:
            return protocol.ResponseWithError(operation=message.get_message(),
                                              error="neural_computation_error",
                                              description="")
        except SliceNotLoadedError:
            return protocol.ResponseWithError(operation=message.get_message(),
                                              error="slice_not_loaded",
                                              description="")
        else:
            axis0, axis1 = res_tensor.shape
            return protocol.ResponsePropagateForward(axis0, axis1, res_tensor.values)


class ClearContextHandler(RequestHandler):
    request_name = "clear_context_request"

    def __call__(self, message):
        try:
            self.context.slice_container.clear_context()
            return protocol.ResponseClearContext()
        except Exception as e:
            return protocol.ResponseWithError(operation=message.get_message(),
                                              error="clear_context_failure",
                                              description=repr(e))