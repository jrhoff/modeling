from pathlib import Path
from dataclasses import dataclass
from typing import Any

class ArgumentParser:

    def __init__(self) -> None:
        pass

    def __call__(self, config_file: Path, *args: Any, **kwds: Any) -> Any:
        pass