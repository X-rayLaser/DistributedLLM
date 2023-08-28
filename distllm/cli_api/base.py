commands = {}


def register_command(cls):
    instance = cls()
    commands[instance.name] = instance


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if cls.__name__ != 'Command':
            register_command(cls)
        return cls


class Command(metaclass=Meta):
    name = ''
    help = ''

    def setup(self, subparsers):
        parser = subparsers.add_parser(self.name, help=self.help)
        self.configure_parser(parser)

    def configure_parser(self, parser):
        pass

    def __call__(self, args):
        pass
