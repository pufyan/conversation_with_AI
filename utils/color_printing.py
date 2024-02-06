import colorama


def _color_print(color, *args, **kwargs):
    try:
        print(color, end='')
        print(*args, **kwargs)
    finally:
        print(colorama.Style.RESET_ALL, end='')


def print_y(*args, **kwargs):
    _color_print(colorama.Fore.YELLOW, *args, **kwargs)


def print_r(*args, **kwargs):
    _color_print(colorama.Fore.RED, *args, **kwargs)


def print_g(*args, **kwargs):
    _color_print(colorama.Fore.GREEN, *args, **kwargs)


def print_pink(*args, **kwargs):
    _color_print(colorama.Fore.MAGENTA, *args, **kwargs)
