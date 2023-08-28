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

    def __call__(self, args):
        run_server(args.host, args.port)
