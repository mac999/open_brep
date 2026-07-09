"""
Entry point for the B-Rep CLI Kernel Modeler.

Usage
    python main.py                 # start the interactive REPL
    python main.py script.bcmd     # run a batch script then drop into the REPL
    python main.py -q script.bcmd  # run a batch script and quit (no REPL)
    python main.py --web           # open the authoring web app, then the REPL
    python main.py --web 9000      # ... on a specific port
"""

import sys

from brep.controller import BRepShell


def _enable_line_buffering() -> None:
    """Make stdout flush each line. Helps on terminals where Python does not
    detect a TTY (e.g. Git Bash / mintty) and would otherwise block-buffer."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass  # stream doesn't support reconfigure; the REPL flushes explicitly


def main(argv=None) -> int:
    _enable_line_buffering()
    argv = list(sys.argv[1:] if argv is None else argv)
    quit_after = False
    if argv and argv[0] in ("-q", "--quit"):
        quit_after = True
        argv = argv[1:]

    # '--web [port]' starts the authoring web app before anything else runs, so
    # a batch script's shapes land in a browser you already have open.
    web_port = None
    if argv and argv[0] in ("--web", "-w"):
        argv = argv[1:]
        web_port = "8765"
        if argv and argv[0].isdigit():
            web_port, argv = argv[0], argv[1:]

    shell = BRepShell()
    if web_port is not None:
        shell.onecmd(f"webapp {web_port}")
    if argv:
        # Run each script file passed on the command line.
        for path in argv:
            shell.onecmd(f'run "{path}"')
        if quit_after:
            return 0
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\ninterrupted - bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
