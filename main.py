from dataclasses import dataclass
from typing import Callable, NoReturn, cast

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from parsing import Chunk, parse
from tui import Sprinkle
from utils import merge


@dataclass
class Args:
    verbose: bool
    output: bool
    editor: bool


async def main(prompt: str, args: Args):
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
  Input: list all files in the current directory including hidden ones
  Output: ls -la

  Input: find all Python files larger than 1MB and sort by size
  Output: find . -name "*.py" -size +1M -exec ls -lh {{}} + | sort -k5 -h

  Input: extract the id property from this json object list
  Output: jq '.[].id'

  Input: extract the number property from this json object list, and sum the result
  Output: jq '.[].number | add'

  Input: create a backup of the config directory with timestamp
  Output: cp -r config config_backup_$(date +%Y%m%d_%H%M%S)

  Input:
  Output: 

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

    chunks, texts = parse(prompt)

    async def invoke(chunk: Chunk) -> Chunk:
        out = await model.ainvoke({"chunk": chunk.text})
        res = Chunk.window(cast(str, out.content), chunk.start, chunk.end)
        if args.verbose:
            print(f"searching for {chunk.text}, got {res.text}")
        return res

    contents = await asyncio.gather(*map(invoke, chunks))

    out = merge(contents, texts, chunk_comparer)
    out = " ".join(o.text for o in out)
    if args.verbose:
        print(f"executing 'bash /usr/env/bash -c {out}")

    def exec(text: str):
        os.execvp("bash", ["/usr/env/bash", "-c", text])

    action = print if args.output else exec

    result = out

    if args.editor:
        app = Sprinkle(out, cast(Callable[[str, Sprinkle], NoReturn], action))
        result = cast(str, await app.run_async())

    action(result)


def chunk_comparer(a: Chunk, b: Chunk) -> int:
    return b.start - a.start


if __name__ == "__main__":
    import argparse
    import asyncio
    import os
    import sys

    parser = argparse.ArgumentParser(description="AI-powered bash command generator")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose output"
    )
    parser.add_argument(
        "-o", "--output", action="store_true", help="output to stdout instead of exec"
    )
    parser.add_argument(
        "-e",
        "--editor",
        action="store_true",
        help="edit the generated script before executing",
    )

    parser.add_argument(
        "prompt", nargs="*", help="natural language description of the command"
    )

    args = parser.parse_args()

    if "GOOGLE_API_KEY" not in os.environ:
        print(
            "\033[91mmissing google genai (gemini) api key. Please set GOOGLE_API_KEY environment variable with a valid gemini api key\033[0",
            file=sys.stderr,
        )
        exit(1)

    prompt = " ".join(args.prompt)
    if len(prompt) == 0:
        print("\033[91mno prompt was provided.\033[0", file=sys.stderr)
        exit(1)

    if args.verbose:
        print(f"Processing prompt: {prompt}", file=sys.stderr)

    asyncio.run(main(prompt, cast(Args, args)))
