import json
from distllm import protocol

routes = {}


def register(message):
    def decorator(func):
        routes[message] = func
        return func

    return decorator


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
        raise NotImplemented


class GetAllSlicesHandler(RequestHandler):
    request_name = "slices_request"

    def __call__(self, message):
        slices = self.get_slices()
        slices_json = json.dumps(slices)
        response = protocol.JsonResponseWithSlices(slices_json)
        response.send(self.context.socket)

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
                self._find_file_and_load_it(slice_name)

                response = protocol.JsonResponseWithLoadedSlice(
                    name=slice_name, model=model
                )
                response.send(self.context.socket)
                return


        # if we got here, that means a given slice wasn't found
        response = protocol.ResponseWithError(operation=message.msg, error='slice_not_found', description='')
        response.send(self.context.socket)

    def _find_file_and_load_it(self, slice_name):
        submission_id = self.context.name_gen.name_to_id(slice_name)
        location = self.context.registry.get_location(submission_id)
        path = location.upload_path
        f = self.context.manager.fs_backend.open_file(path, mode='rb')
        load_slice(f)
        f.close()


def load_slice(f):
    """load_slice in memory"""


class FileSubmissionBeginHandler(RequestHandler):
    request_name = "request_file_submission_begin"

    def __call__(self, message):
        metadata = json.loads(message.metadata_json)
        submission_id = self.context.manager.prepare_upload(metadata)


class SubmitPartHandler(RequestHandler):
    request_name = "request_submit_part"

    def __call__(self, message):
        num_bytes = self.manager.upload_part(message.submission_id, message.data)


class FileSubmissionEndHandler(RequestHandler):
    request_name = "request_file_submission_end"

    def __call__(self, message):
        from distllm import compute_node

        try:
            total_size = self.manager.finilize_upload(message.submission_id, message.checksum)
        except compute_node.FailedUploadError:
            # send corresponding response
            pass
        else:
            name = self.name_gen.id_to_name(message.submission_id)
