"""Make the repo root importable for tests run from anywhere (so `import config`,
`from envs... import` resolve without an editable install)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
