from dataclasses import dataclass
import json
import os
import hashlib

from distllm.utils import FakeFileSystemBackend


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


@dataclass
class UploadLocation:
    dir_path: str
    upload_path: str
    metadata_path: str


class UploadRegistry:
    registry_data_file = "registry_data.json"
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
        return UploadLocation(base_path, upload_path, metadata_path)

    @classmethod
    def registry_data_path(cls, root):
        return os.path.join(root, cls.registry_data_file)

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, state_json):
        state = json.loads(state_json)
        root = state['root']
        obj = cls(root)
        obj.__dict__ = state
        return obj

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict.copy()
        self.id_to_metadata = {int(idx): meta for idx, meta in self.id_to_metadata.items()}

    def _check_index(self, submission_id):
        all_uploads = self.in_progress + self.failed + self.finished
        try:
            all_uploads.index(submission_id)
        except ValueError:
            raise UploadNotFoundError

    def _check_active_upload(self):
        if not self.in_progress:
            raise NoActiveUploadError


class UploadManager:
    def __init__(self, registry):
        self.registry = registry
        self.id_to_upload = {}  # maps submission_id to active uploads
        self.fs_backend = FakeFileSystemBackend()

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
            save_path = self.registry.registry_data_path(self.registry.root)
            f = self.fs_backend.open_file(save_path, "w")
            state_dict = self.registry.state_dict()
            state_json = json.dumps(state_dict)
            f.write(state_json)
            f.close()

    def get_upload(self, submission_id):
        upload = self.id_to_upload.get(submission_id)
        if not upload:
            raise FileNotFoundError
        return upload


class ParallelUploadError(Exception):
    pass


class UploadNotFoundError(Exception):
    pass


class FailedUploadError(Exception):
    pass


class NoActiveUploadError(Exception):
    pass


class FunkyNameGenerator:
    def __init__(self, names):
        self.names = names

    def id_to_name(self, submission_id):
        try:
            return self.names[submission_id]
        except IndexError:
            return None

    def name_to_id(self, name):
        try:
            return self.names.index(name)
        except ValueError:
            return None


# singleton objects persistent in memory
upload_registry = UploadRegistry('uploads')
upload_manager = UploadManager(upload_registry)
