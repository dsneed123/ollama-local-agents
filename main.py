#!/usr/bin/env python3
"""Easy startup script: ``python main.py`` launches the local-agents CLI.

Equivalent to running ``local-agents``. Works from a fresh clone without
installing the package, as long as the dependencies (httpx, pyyaml) are
available. All CLI arguments are passed through, e.g.::

    python main.py                       # interactive menu
    python main.py list                  # list agents
    python main.py run coding "fix the failing test"
    python main.py --debug
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

try:
    from local_agents.cli.main import main
except ModuleNotFoundError:
    # Not installed: fall back to the in-repo source tree.
    sys.path.insert(0, str(SRC))
    try:
        from local_agents.cli.main import main
    except ModuleNotFoundError as exc:
        missing = exc.name or "a dependency"
        sys.stderr.write(
            f"Could not import local-agents ({missing} is missing).\n"
            "Install the project first:\n\n"
            "    pip install -e .\n\n"
        )
        raise SystemExit(1) from exc

if __name__ == "__main__":
    raise SystemExit(main())
