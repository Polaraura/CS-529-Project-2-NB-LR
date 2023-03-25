# FIXME: moved create_sub_directories() here due to circular import...
from pathlib import Path


def create_sub_directories(filepath: str):
    # recursively create sub directories
    filepath_path = Path(filepath)
    filepath_path.mkdir(parents=True, exist_ok=True)
