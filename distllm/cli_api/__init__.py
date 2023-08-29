import argparse

from .base import commands
from . import (provision, run_node, status, push_slice, load_slice)

def parse_all_args():
    root_parser = argparse.ArgumentParser(
        description='Utilities to work with distributed LLM'
    )
    all_subparsers = root_parser.add_subparsers(dest='command')

    for name, command in commands.items():
        command.setup(all_subparsers)

    return root_parser.parse_args()


def execute_command():
    args = parse_all_args()
    command = commands[args.command]
    command(args)