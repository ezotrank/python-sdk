"""Concrete resource implementations."""

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Union

import anyio
import anyio.to_thread
import httpx
from pydantic import Field, validator
from pydantic.json import pydantic_encoder

from mcp.server.fastmcp.resources.base import Resource


class TextResource(Resource):
    """A resource that reads from a string."""

    text: str = Field(description="Text content of the resource")

    async def read(self) -> str:
        """Read the text content."""
        return self.text


class BinaryResource(Resource):
    """A resource that reads from bytes."""

    data: bytes = Field(description="Binary content of the resource")

    async def read(self) -> bytes:
        """Read the binary content."""
        return self.data


class FunctionResource(Resource):
    """A resource that defers data loading by wrapping a function.

    The function is only called when the resource is read, allowing for lazy loading
    of potentially expensive data. This is particularly useful when listing resources,
    as the function won't be called until the resource is actually accessed.

    The function can return:
    - str for text content (default)
    - bytes for binary content
    - other types will be converted to JSON
    """

    fn: Callable[[], Any] = Field(exclude=True)

    async def read(self) -> Union[str, bytes]:
        """Read the resource by calling the wrapped function."""
        try:
            result = (
                await self.fn() if inspect.iscoroutinefunction(self.fn) else self.fn()
            )
            if isinstance(result, Resource):
                read_result = await result.read()
                # Ensure the result of reading the nested resource is str or bytes
                if isinstance(read_result, (str, bytes)):
                    return read_result
                else:
                    # Fallback: attempt JSON serialization or string conversion
                    try:
                        return json.dumps(read_result, default=pydantic_encoder)
                    except TypeError:
                        return str(read_result)
            if isinstance(result, bytes):
                return result
            if isinstance(result, str):
                return result
            # For other types, attempt JSON serialization
            try:
                return json.dumps(result, default=pydantic_encoder)
            except TypeError:
                # If JSON serialization fails, try str()
                return str(result)
        except Exception as e:
            raise ValueError(f"Error reading resource {self.uri}: {e}")


class FileResource(Resource):
    """A resource that reads from a file.

    Set is_binary=True to read file as binary data instead of text.
    """

    path: Path = Field(description="Path to the file")
    is_binary: bool = Field(
        default=False,
        description="Whether to read the file as binary data",
    )
    mime_type: str = Field(
        default="text/plain",
        description="MIME type of the resource content",
    )

    @validator("path")
    @classmethod
    def validate_absolute_path(cls, path: Path) -> Path:
        """Ensure path is absolute."""
        if not path.is_absolute():
            raise ValueError("Path must be absolute")
        return path

    @validator("is_binary", always=True) # always=True needed to access other fields via `values`
    @classmethod
    def set_binary_from_mime_type(cls, is_binary: bool, values: Dict[str, Any]) -> bool:
        """Set is_binary based on mime_type if not explicitly set."""
        # If is_binary is already True (explicitly set or default), keep it.
        # Or if mime_type isn't available yet in validation context (shouldn't happen with always=True)
        if is_binary or "mime_type" not in values:
             return is_binary # Return the provided value

        # If is_binary is False (default), derive from mime_type
        mime_type = values.get("mime_type", "text/plain")
        # Consider it binary if mime_type is known and doesn't start with "text/"
        is_likely_binary = isinstance(mime_type, str) and not mime_type.startswith("text/")
        return is_likely_binary


    async def read(self) -> Union[str, bytes]:
        """Read the file content."""
        try:
            if self.is_binary:
                return await anyio.to_thread.run_sync(self.path.read_bytes)
            return await anyio.to_thread.run_sync(self.path.read_text)
        except Exception as e:
            raise ValueError(f"Error reading file {self.path}: {e}")


class HttpResource(Resource):
    """A resource that reads from an HTTP endpoint."""

    url: str = Field(description="URL to fetch content from")
    mime_type: str = Field(
        default="application/json", description="MIME type of the resource content"
    )

    async def read(self) -> Union[str, bytes]:
        """Read the HTTP content."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.url)
                response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
                # Heuristic to decide whether to return text or bytes based on mime_type
                content_type = response.headers.get("content-type", self.mime_type).lower()
                if content_type.startswith("text/"):
                    return response.text
                elif content_type == "application/json":
                     # Even if JSON, return as text for consistency unless specifically binary
                     return response.text
                else:
                    # Assume binary for other types
                    return response.content
            except httpx.HTTPStatusError as e:
                 raise ValueError(f"HTTP error fetching {self.url}: {e.response.status_code} {e.response.reason_phrase}") from e
            except httpx.RequestError as e:
                 raise ValueError(f"Error requesting {self.url}: {e}") from e


class DirectoryResource(Resource):
    """A resource that lists files in a directory."""

    path: Path = Field(description="Path to the directory")
    recursive: bool = Field(
        default=False, description="Whether to list files recursively"
    )
    pattern: Union[str, None] = Field( # Use Union for Optional in Pydantic v1
        default=None, description="Optional glob pattern to filter files"
    )
    mime_type: str = Field(
        default="application/json", description="MIME type of the resource content"
    )

    @validator("path")
    @classmethod
    def validate_absolute_path(cls, path: Path) -> Path:
        """Ensure path is absolute."""
        if not path.is_absolute():
            raise ValueError("Path must be absolute")
        return path

    def list_files(self) -> list[Path]:
        """List files in the directory."""
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: {self.path}")
        if not self.path.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.path}")

        try:
            glob_method = self.path.rglob if self.recursive else self.path.glob
            pattern_to_use = self.pattern if self.pattern else "*"
            return list(glob_method(pattern_to_use))
        except Exception as e:
            raise ValueError(f"Error listing directory {self.path} with pattern '{self.pattern}': {e}") from e

    async def read(self) -> str:  # Always returns JSON string
        """Read the directory listing."""
        try:
            # Run the synchronous list_files in a thread
            all_paths = await anyio.to_thread.run_sync(self.list_files)
            # Filter for files and make paths relative to the resource's root path
            file_list = [str(p.relative_to(self.path)) for p in all_paths if p.is_file()]
            return json.dumps({"files": file_list}, indent=2)
        except Exception as e:
            # Catch potential errors during listing or JSON dumping
            raise ValueError(f"Error reading directory listing for {self.path}: {e}") from e
