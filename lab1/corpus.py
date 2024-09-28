from pathlib import Path


def read_all_files(directory):
    files = []
    for file in directory.iterdir():
        if file.is_file():
            with open(file, "r") as f:
                files.append(f.read())
    return files


dir = Path("lab1/corpus")

corpus = read_all_files(dir)
