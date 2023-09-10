from .base import Command
from ..proxy_node import run_proxy


class RunCommand(Command):
    name = 'run_proxy'
    help = 'Deploy a proxy node on this device and connect to a compute node run with --reverse option'

    def configure_parser(self, parser):
        parser.add_argument('--host', type=str, default='localhost',
                            help='Proxy IP address')

        parser.add_argument('--client-port', type=int, default=9996,
                            help='Proxy port number for serving for a client')

        parser.add_argument('--node-port', type=int, default=9997,
                            help='Proxy port number for serving for a compute node')

    def __call__(self, args):
        run_proxy(args.host, args.client_port, args.node_port)
