"""`typed-decisions` command: one entry point for the server and the distillation tools.

    typed-decisions serve --preset apple
    typed-decisions distill train --log traffic.jsonl --family <key> --out heads/
    typed-decisions version
"""

from __future__ import annotations

import sys


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else "help"
    if command == "serve":
        from .server import main as serve
        return serve(argv)
    if command == "distill":
        from .distill import main as distill
        return distill(argv)
    if command == "version":
        from importlib.metadata import version
        print(version("typed-decisions"))
        return None
    print(__doc__.strip())
    if command not in ("help", "-h", "--help"):
        raise SystemExit(f"unknown command {command!r}")
    return None


if __name__ == "__main__":
    main()
