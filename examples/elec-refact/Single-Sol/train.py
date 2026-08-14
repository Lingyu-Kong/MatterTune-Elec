from __future__ import annotations

import sys
from pathlib import Path


SCENARIO_ROOT = Path(__file__).resolve().parent
COMMON_ROOT = SCENARIO_ROOT.parent
sys.path.insert(0, str(COMMON_ROOT))

from training import run_from_cli  # noqa: E402


if __name__ == "__main__":
    run_from_cli(SCENARIO_ROOT / "configs" / "default.yml")
