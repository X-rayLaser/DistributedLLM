from .base import Command
from ..control_center import Connection


class StatusCommand(Command):
    name = 'status'
    help = 'Get status of a whole compute network or just a single node'

    def configure_parser(self, parser):
        parser.add_argument('--address', type=str, default='',
                            help='IP address of a compute node in the format "IP:PORT".')

    def __call__(self, args):

        if args.address:
            host, port = args.address.split(":")

            connection = Connection((host, int(port)))
            status = connection.get_status()
            print(status)
        else:
            print("Not implemented yet")
