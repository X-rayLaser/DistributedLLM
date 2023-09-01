from .base import Command
from ..control_center import Connection


class ListSlicesCommand(Command):
    name = 'list_slices'
    help = 'Show a list of all slices uploaded to a given node'

    def configure_parser(self, parser):
        parser.add_argument('address', type=str,
                            help='IP address of a compute node in the format "IP:PORT".')

    def __call__(self, args):
        host, port = args.address.split(":")

        connection = Connection((host, int(port)))
        slices = connection.list_all_slices()
        print(slices)
