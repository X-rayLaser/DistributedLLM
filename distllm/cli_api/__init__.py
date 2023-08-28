import argparse

from .base import commands
from . import (provision, run_node)

def parse_all_args():
    root_parser = argparse.ArgumentParser(
        description='A set of tools for training a neural net for handwriting recognition'
    )
    all_subparsers = root_parser.add_subparsers(dest='command')

    for name, command in commands.items():
        command.setup(all_subparsers)

    return root_parser.parse_args()


def execute_command():
    args = parse_all_args()
    command = commands[args.command]
    command(args)