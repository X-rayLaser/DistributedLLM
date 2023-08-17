import hashlib
import os
import json
from dataclasses import dataclass
from pathlib import Path
import io
from distllm import protocol
from distllm.protocol import receive_message, restore_message


class TCPHandler:
    def __init__(self, socket):
        self.registry = UploadRegistry('uploads')
        self.manager = UploadManager(self.registry)
        self.name_gen = FunkyNameGenerator()
        self.socket = socket

    def handle(self):
        msg, body = receive_message(self.socket)
        message = restore_message(msg, body)
        if message.get_message() == "slices_request":
            ids = self.registry.finished

            slices = []
            for submission_id in ids:
                location = self.registry.get_location(submission_id)
                path = location.metadata_path
                f = self.manager.fs_backend.open_file(path, 'r')
                s = f.read()
                f.close()
                metadata = json.loads(s)
                if metadata['type'] != 'slice':
                    continue

                model = metadata['model']
                layer_from = metadata['layer_from']
                layer_to = metadata['layer_to']
                slice_name = self.name_gen.id_to_name[submission_id]
                slices.append(dict(name=slice_name, model=model,
                                   layer_from=layer_from, layer_to=layer_to))
            
            slices_json = json.dumps(slices)
            response = protocol.JsonResponseWithSlices(slices_json)
            response.send(self.socket)
        return

        if self.prepare_msg():
            metadata = self.get_metadata()
            submission_id = self.manager.prepare_upload(metadata)
        elif self.upload_part():
            submission_id, blob = self.get_part()

            num_bytes = self.manager.upload_part(submission_id, blob)
        elif self.finilize():
            submission_id, checksum = self.get_submission_id_and_checksum()

            try:
                total_size = self.manager.finilize_upload(submission_id, checksum)
            except FailedUploadError:
                # send corresponding response
                pass
            else:
                name = self.name_gen.id_to_name(submission_id)


class UploadManager:
    def __init__(self, registry):
        self.registry = registry
        self.id_to_upload = {}  # maps submission_id to active uploads
        self.fs_backend = DefaultFileSystemBackend()

    def prepare_upload(self, metadata) -> int:
        submission_id = self.registry.add_upload(metadata)
        upload_location = self.registry.get_location(submission_id)

        file_handler = self._prepare_file_tree(upload_location, metadata)
        self.id_to_upload[submission_id] = FileUpload(file_handler)
        return submission_id

    def _prepare_file_tree(self, upload_location, metadata):
        self.fs_backend.make_dirs(upload_location.dir_path, exists_ok=True)

        f = self.fs_backend.open_file(upload_location.metadata_path, "w")
        s = json.dumps(metadata)
        f.write(s)
        f.close()

        return self.fs_backend.open_file(upload_location.upload_path, "wb")

    def upload_part(self, submission_id, blob) -> int:
        upload = self.id_to_upload.get(submission_id)
        if not upload:
            raise UploadNotFoundError
        
        upload.update(blob)
        return len(blob)

    def finilize_upload(self, submission_id, checksum) -> int:
        upload = self.get_upload(submission_id)

        try:
            upload.validate(checksum)
            self.registry.mark_finished()
            return upload.total_size()
        except FailedUploadError:
            self.registry.mark_failed()
            raise
        finally:
            upload.f.close()

    def get_upload(self, submission_id):
        upload = self.id_to_upload.get(submission_id)
        if not upload:
            raise FileNotFoundError
        return upload


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


class FunkyNameGenerator:
    def id_to_name(self, submission_id):
        pass

    def name_to_id(self, name):
        pass


class UploadRegistry:
    slices_dir = "slices"
    other_dir = "other_files"

    uploaded_file = "uploaded_file"
    metadata_file = "metadata.json"

    def __init__(self, root):
        self.root = root
        self.in_progress = []  # current active uploads
        self.finished = []
        self.failed = []
        self.num_uploads = 0
        self.id_to_metadata = {}

    def add_upload(self, metadata):
        if self.in_progress:
            raise ParallelUploadError

        upload_id = self.num_uploads
        self.num_uploads += 1
        self.in_progress.append(upload_id)
        self.id_to_metadata[upload_id] = metadata
        return upload_id

    def mark_finished(self):
        self._check_active_upload()
        self.finished.append(self.in_progress.pop())
        
    def mark_failed(self):
        self._check_active_upload()
        self.failed.append(self.in_progress.pop())

    def get_location(self, submission_id):
        self._check_index(submission_id)

        metadata = self.id_to_metadata[submission_id]
        if metadata['type'] == 'slice':
            sub_dir = self.slices_dir
        else:
            sub_dir = self.other_dir
    
        base_path = os.path.join(self.root, sub_dir, f'upload_{submission_id}')
        upload_path = os.path.join(base_path, self.uploaded_file)
        metadata_path = os.path.join(base_path, self.metadata_file)
        return UploadLocation(self.root, upload_path, metadata_path)

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, state_json):
        state = json.loads(state_json)
        root = state['root']
        obj = cls(root)
        obj.__dict__ = state
        return obj

    def _check_index(self, submission_id):
        try:
            self.in_progress.index(submission_id)
        except ValueError:
            raise UploadNotFoundError

    def _check_active_upload(self):
        if not self.in_progress:
            raise NoActiveUploadError


@dataclass
class UploadLocation:
    dir_path: str
    upload_path: str
    metadata_path: str


class FileUpload:
    def __init__(self, f):
        self.f = f
        self.byte_count = 0
        self.hasher = hashlib.sha256()

    def update(self, blob):
        pos = 0

        while pos < len(blob):
            pos += self.f.write(blob[pos:])

        self.byte_count += len(blob)
        self.hasher.update(blob)

    def checksum(self):
        return self.hasher.hexdigest()

    def total_size(self):
        return self.byte_count

    def validate(self, target_checksum):
        if self.checksum() != target_checksum:
            raise FailedUploadError


class ParallelUploadError(Exception):
    pass


class UploadNotFoundError(Exception):
    pass


class FailedUploadError(Exception):
    pass


class NoActiveUploadError(Exception):
    pass
