import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_and_mkdir_if_neccasiry(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path