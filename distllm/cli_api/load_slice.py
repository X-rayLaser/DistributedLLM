from .base import Command
from ..control_center import Connection


class LoadSliceCommand(Command):
    name = 'load_slice'
    help = 'Load model slice in memory on a specified node'

    def configure_parser(self, parser):
        parser.add_argument('address', type=str,
                            help='IP address of a compute node in the format "IP:PORT".')

        parser.add_argument('name', type=str,
                            help='Name assigned to the slice')

    def __call__(self, args):
        host, port = args.address.split(":")

        connection = Connection((host, int(port)))
        res = connection.load_slice(args.name)
        print(res)
