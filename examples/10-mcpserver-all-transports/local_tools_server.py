#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Local Dev Tools MCP Server (stdio transport)
=============================================

A lightweight MCP server that exposes utility tools over stdio.  The Dapr
sidecar spawns this process and communicates via stdin/stdout when the
``local-tools`` MCPServer resource is loaded.

This server is referenced by ``resources/local-tools.yaml``::

    endpoint:
      stdio:
        command: python
        args: ["-m", "local_tools_server"]

Tools exposed:
  - ``search_files``: Search for files by name pattern.
  - ``summarize_text``: Produce a brief summary of the given text.
"""

import os
import fnmatch
import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("local-tools-server")

mcp = FastMCP("LocalDevTools")


@mcp.tool()
async def search_files(directory: str, pattern: str) -> str:
    """Search for files matching a glob pattern in the given directory.

    Args:
        directory: Absolute path to the directory to search.
        pattern: Glob pattern to match file names (e.g. '*.py', 'test_*').

    Returns:
        Newline-separated list of matching file paths, or a message if none found.
    """
    matches: list[str] = []
    if not os.path.isdir(directory):
        return f"Error: '{directory}' is not a valid directory."

    for root, _dirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(root, name))
        if len(matches) >= 50:
            break

    if not matches:
        return f"No files matching '{pattern}' found in {directory}."
    return "\n".join(matches[:50])


@mcp.tool()
async def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Produce a brief extractive summary by returning the first N sentences.

    Args:
        text: The text to summarize.
        max_sentences: Maximum number of sentences to include (default 3).

    Returns:
        The first ``max_sentences`` sentences of the input text.
    """
    sentences: list[str] = []
    for part in text.replace("\n", " ").split("."):
        stripped = part.strip()
        if stripped:
            sentences.append(stripped + ".")
        if len(sentences) >= max_sentences:
            break
    return " ".join(sentences) if sentences else text[:200]


if __name__ == "__main__":
    mcp.run("stdio")
