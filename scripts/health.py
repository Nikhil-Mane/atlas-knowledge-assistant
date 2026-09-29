"""Check every backing service and the settings that matter for throughput.

Exit code 0 when everything is OK, 1 otherwise.
Usage:  .venv/Scripts/python scripts/health.py
"""
import sys

from rag.config import get_settings
from rag.health import run_checks


def main() -> int:
    results = run_checks(get_settings())
    for name, r in results.items():
        print(f"{'OK  ' if r['ok'] else 'FAIL'}  {name:9} {r['detail']}")
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
