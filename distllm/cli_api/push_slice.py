import json
import os

from .base import Command
from ..control_center import Connection


class PushSliceCommand(Command):
    name = 'push_slice'
    help = 'Push a slice of model layers to a specified compute node'

    def configure_parser(self, parser):
        parser.add_argument('address', type=str,
                            help='IP address of a compute node in the format "IP:PORT".')

        parser.add_argument('slice', type=str,
                            help='Path to the slice')

        parser.add_argument('metadata', type=str,
                            help='Json formatted file containing additional details')

    def __call__(self, args):
        host, port = args.address.split(":")

        connection = Connection((host, int(port)))

        with open(args.metadata) as f:
            metadata = json.loads(f.read())

        # todo: fix this after removing model argument from push_slice method

        model_id = metadata['model_id']

        del metadata['model_id']

        file_size = os.path.getsize(args.slice)
        with open(args.slice, "rb") as f:
            res = connection.push_slice(f, model_id, metadata,
                                        file_size=file_size, progress_bar=True)
            print(res)
