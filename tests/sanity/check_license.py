# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from argparse import ArgumentParser
from collections.abc import Iterable
from pathlib import Path

LICENSE_HEADER_RE = re.compile(
    r"Copyright \d{4}(?:-\d{4})? (?:Huawei Technologies Co\., Ltd\. All Rights Reserved\.|The TransferQueue Team)"
)
SUPPORTED_SUFFIXES = {".py", ".rs"}
IGNORED_DIRS = {
    "__pycache__",
    "build",
    "dist",
    "target",
    "venv",
    "worktrees",
}


def get_source_files(path_arg: Path) -> Iterable[Path]:
    """Yield Python and Rust source files under a directory or file path."""
    if path_arg.is_file():
        if path_arg.suffix in SUPPORTED_SUFFIXES:
            yield path_arg
        return
    if not path_arg.is_dir():
        return

    for root, dirs, files in os.walk(path_arg):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in IGNORED_DIRS and not d.endswith(".egg-info")]
        root_path = Path(root)
        for file in files:
            candidate = root_path / file
            if candidate.suffix in SUPPORTED_SUFFIXES:
                yield candidate


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--directories",
        "-d",
        required=True,
        type=Path,
        nargs="+",
        help="List of directories or files to check for license headers",
    )
    args = parser.parse_args()

    pathlist = sorted({path.resolve() for path_arg in args.directories for path in get_source_files(path_arg)})

    for path in pathlist:
        path_in_str = str(path)
        print(path_in_str)
        header_text = "\n".join(path.read_text(encoding="utf-8").splitlines()[:20])
        assert LICENSE_HEADER_RE.search(header_text), f"file {path_in_str} does not contain license"
