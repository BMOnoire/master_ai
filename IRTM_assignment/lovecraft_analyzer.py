import os
from pathlib import Path
from os import listdir
from os.path import isfile, join
import docx2txt
import re

path = Path("./fictions/")


def main():
    file_list = [f for f in listdir(path) if isfile(join(path, f))]

    # extract text
    book_list = []
    for file_name in file_list:
        correct_regex = re.match("\A\((\d*)\)\[(\d*)\] (.*)\.docx\Z",file_name)
        if not correct_regex:
            print(file_name, " is not loaded (regex problem)")
            return 1

        book = {
            "id"    : int(correct_regex.group(1)),
            "year"  : int(correct_regex.group(2)),
            "title" : correct_regex.group(3),
            "text"  : docx2txt.process(path / file_name)
        }
        book_list.append(book)

    def sortLogic(obj):
        return obj["id"]
    book_list.sort(key = sortLogic)

main()