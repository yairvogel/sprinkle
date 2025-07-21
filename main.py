from typing import Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    prompt: str
    start: int
    end: int


def main(prompt: str):
    chunks: list[Chunk] = []
    start = prompt.find("{{")
    while 0 <= start < len(prompt):
        end = prompt.find("}}", start)
        if end < 0:
            break
        chunks.append(Chunk(prompt, start, end))
        start = prompt.find("{{", end)

    print([c.prompt[c.start + 2 : c.end] for c in chunks])


if __name__ == "__main__":
    import sys

    argv = sys.argv
    main(str.join(" ", argv[1:]))
