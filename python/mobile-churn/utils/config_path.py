import os
import re
import sys

REGEX = "^[v][0-9]+$"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def print_dir_list(dir_name_list):
    for i, name in enumerate(dir_name_list):
        print(f"  [{i}] {name}")


def config_dir_list(path) -> list:
    try:
        subdirs = [
            name
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
            and compare_pattern(name, REGEX)
        ]
        subdirs_sorted = sorted(
            subdirs, 
            key=lambda x: int(re.sub(r'\D', '', x)) 
        )
        
        return subdirs_sorted
        
    except FileNotFoundError:
        print(f"Error: the path '{path}' does not exist")
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
