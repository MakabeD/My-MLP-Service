import os
import re
import sys

REGEX = "^[v][0-9]+$"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def print_dir_list(dir_name_list):
    for i, name in enumerate(dir_name_list):
        print(f"  [{i}] {name}")


def config_dir_list(path):
    try:
        subdirectorios = [
            nombre
            for nombre in os.listdir(path)
            if os.path.isdir(os.path.join(path, nombre))
            and compare_pattern(nombre, REGEX)
        ]
        return subdirectorios
    except FileNotFoundError:
        print(f"Error: the path '{path}' do not exist")
        return []
    except PermissionError:
        print(f"Error: do not have access to '{path}'")
        return []


def compare_pattern(string, pattern):
    try:
        regex = re.compile(pattern)
    except re.error as e:
        print(f"Error at pattern: {e}")
        return False
    return bool(regex.fullmatch(string))  # '^[v][0-9]+$'print(list_dir('./configs'))
