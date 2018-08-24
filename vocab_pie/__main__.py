import os
import sys
from typing import List

from configargparse import ArgumentParser

DEFAULT_IGNORE_CASE = True


def __main():
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--ignore-case", type=bool, default=DEFAULT_IGNORE_CASE)
    args = parser.parse_args()

    try:
        create_from_file(args.file)
    except VocabPieError as e:
        e.display()


def create_from_file(file_path: str, ignore_case: bool = DEFAULT_IGNORE_CASE):
    if not os.path.isfile(file_path):
        raise VocabPieError(f"File does not exist: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise VocabPieError(f"Can not read file: {file_path}")

    with open(file_path, "r") as f:
        sentences = [line.strip() for line in f if line != ""]

    create_from_sentences(sentences, ignore_case)


def create_from_sentences(
        sentences: List[str],
        ignore_case: bool = DEFAULT_IGNORE_CASE):

    if ignore_case:
        sentences = [s.lower() for s in sentences]

    print(sentences)
    print(ignore_case)


class VocabPieError(BaseException):
    def __init__(self, message: str):
        self.__message = message

    def display(self):
        print("Error: " + self.__message, file=sys.stderr)


if __name__ == "__main__":
    __main()
