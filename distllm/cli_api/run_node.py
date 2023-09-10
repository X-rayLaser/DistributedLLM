from .base import Command
from ..compute_node.serve import run_server


class RunCommand(Command):
    name = 'run_node'
    help = 'Deploy a compute node on this device'

    def configure_parser(self, parser):
        parser.add_argument('--host', type=str, default='localhost',
                            help='Host IP address')

        parser.add_argument('--port', type=int, default=9999,
                            help='Port number')

        parser.add_argument('--uploads_dir', type=str, default='uploads',
                            help='Home for all uploaded files and model slices')

        parser.add_argument('--reverse', type=bool, default=False, action='store_true',
                            help='Connect to a given address from here, then serve')

    def __call__(self, args):
        run_server(args.host, args.port, args.uploads_dir, args.reverse)
