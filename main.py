from dataclasses import dataclass
from typing import Iterable, Protocol, TypeVar, cast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


@dataclass
class Chunk:
    prompt: str
    start: int
    end: int
    full: bool = False

    @property
    def text(self) -> str:
        return self.prompt if self.full else self.prompt[self.start : self.end]

    def __le__(self, other):
        return self.start <= other.start


async def main(prompt: str):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=1000,
        timeout=None,
        max_retries=2,
    )

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a bash command generator AI. Your role is to convert natural language descriptions of actions into syntactically correct bash commands that execute those descriptions entirely.

  ## Core Functionality
  - Take free text descriptions of system operations, file manipulations, or administrative tasks
  - Output only the corresponding bash command(s) that would execute the described action
  - Ensure commands are syntactically correct and executable
  - Handle complex multi-step operations by chaining commands appropriately

  ## Guidelines
  1. **Output Format**: Provide only the bash command(s), no explanations unless explicitly requested
  2. **Completeness**: The command must fully accomplish the described action
  3. **Safety**: Include appropriate error handling and safety flags where applicable
  4. **Efficiency**: Use the most direct and efficient command structure
  5. **Portability**: Prefer POSIX-compatible commands when possible, use only gnu tools

  ## Examples
  Input: "list all files in the current directory including hidden ones"
  Output: `ls -la`

  Input: find all Python files larger than 1MB and sort by size
  Output: find . -name "*.py" -size +1M -exec ls -lh {{}} + | sort -k5 -h

  Input: create a backup of the config directory with timestamp
  Output: cp -r config config_backup_$(date +%Y%m%d_%H%M%S)

  Input: ""

  ## Command Chaining
  - Use `&&` for sequential execution with failure stops
  - Use `;` for sequential execution regardless of failures
  - Use `|` for piping output between commands
  - Use `||` for fallback operations

  ## Error Handling
  - Include appropriate flags like `-f` for force operations when the description implies overwriting
  - Use error redirection (`2>/dev/null`) when the description suggests ignoring errors
  - Add confirmation prompts (`-i`) for destructive operations unless explicitly bypassed

  Focus on accuracy and completeness. The generated command should execute the described action entirely without requiring additional manual steps.
""",
            ),
            ("human", "{chunk}"),
        ]
    )

    model = template | model

    chunks: list[Chunk] = []
    start = prompt.find("{{")
    while 0 <= start < len(prompt):
        end = prompt.find("}}", start)
        if end < 0:
            break
        chunks.append(Chunk(prompt, start, end + 2))
        start = prompt.find("{{", end)

    texts: list[Chunk] = []
    if chunks[0].start > 0:
        texts.append(Chunk(prompt, 0, chunks[0].start))
    for p, n in zip(chunks, chunks[1:]):
        texts.append(Chunk(prompt, p.end, n.start))
    if chunks[-1].end < len(prompt):
        texts.append(Chunk(prompt, chunks[-1].end, len(prompt)))

    contents = []

    async def invoke(chunk: Chunk) -> Chunk:
        print(f"hi {chunk.start}")
        out = await model.ainvoke({"chunk": chunk.text})
        res = Chunk(cast(str, out.content), chunk.start, chunk.end, full=True)
        print(f"bi {chunk.start}")
        return res

    contents = await asyncio.gather(*(invoke(c) for c in chunks))

    out = merge(contents, texts)
    print(" ".join(o.text for o in out))


class Comparable(Protocol):
    def __le__(self, other) -> bool: ...


T = TypeVar("T", bound=Comparable)


def merge(L: list[T], R: list[T]) -> Iterable[T]:
    i = 0  # Initial index of first subarray
    j = 0  # Initial index of second subarray
    n1 = len(L)
    n2 = len(R)

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            yield L[i]
            i += 1
        else:
            yield R[j]
            j += 1

    # Copy the remaining elements of L[], if there
    # are any
    while i < n1:
        yield L[i]
        i += 1

    # Copy the remaining elements of R[], if there
    # are any
    while j < n2:
        yield R[j]
        j += 1


if __name__ == "__main__":
    import os
    import sys

    if "GOOGLE_API_KEY" not in os.environ:
        print(
            "\033[91mmissing google genai (gemini) api key. Please set GOOGLE_API_KEY environment variable with a valid gemini api key\033[0",
            file=sys.stderr,
        )
        exit(1)

    argv = sys.argv
    import asyncio

    asyncio.run(main(str.join(" ", argv[1:])))
