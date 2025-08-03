from dataclasses import dataclass
from typing import Callable, NoReturn, cast


from inference import create_template
from parsing import Chunk, parse
from tui import Sprinkle
from utils import merge


@dataclass
class Args:
    verbose: bool
    output: bool
    editor: bool


async def main(prompt: str, args: Args):
    chunks, texts = parse(prompt)

    template = create_template()

    async def invoke(chunk: Chunk) -> Chunk:
        prompt_replaced = prompt.replace(chunk.text, "YOUR ANSWER IS HERE")
        out = await template.ainvoke({"chunk": chunk.text, "text": prompt_replaced})
        out = cast(str, out.content)
        out = out.strip("\"'")
        res = Chunk.window(out, chunk.start, chunk.end)
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
