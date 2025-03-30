"""Base classes and interfaces for FastMCP resources."""

import abc
from typing import Optional, Union, Dict, Any

from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    validator,
)


class Resource(BaseModel, abc.ABC):
    """Base class for all resources."""

    class Config:
        validate_assignment = True
        # Pydantic V1 defaults to allowing extra fields, so no need for `extra = Extra.allow` unless we want to be explicit or change it later

    uri: AnyUrl = Field(
        ..., description="URI of the resource" # Ellipsis (...) makes it required in V1
    )
    name: Optional[str] = Field(description="Name of the resource", default=None)
    description: Optional[str] = Field(
        description="Description of the resource", default=None
    )
    mime_type: str = Field(
        default="text/plain",
        description="MIME type of the resource content",
        regex=r"^[a-zA-Z0-9]+/[a-zA-Z0-9\-+.]+$", # Use regex instead of pattern in V1
    )

    @validator("name", pre=True, always=True)
    @classmethod
    def set_default_name(cls, name: Optional[str], values: Dict[str, Any]) -> str:
        """Set default name from URI if not provided."""
        if name:
            return name
        uri = values.get("uri")
        if uri:
            # In Pydantic V1, AnyUrl might already be parsed at this stage if pre=True
            # Ensure it's a string for the name field
            return str(uri)
        # If 'uri' itself is missing (which it shouldn't be, as it's required),
        # Pydantic's default validation will catch it before this validator runs usually.
        # But if uri was None or '', this logic path could be hit.
        # The original check ensured either name or uri exists; URI is mandatory now.
        # If name is None and URI *is* present (as required), we return str(uri).
        # This validator primarily handles the case where name is *not* provided.
        # If we reach here, it means 'name' was not provided AND 'uri' was not found in values,
        # which is an inconsistent state given 'uri' is required. Pydantic V1's required field
        # validation should ideally prevent this. Raising an error might be redundant
        # but keeps the original intent if somehow validation reaches here unexpectedly.
        raise ValueError("URI is required to set a default name when name is not provided.")


    @abc.abstractmethod
    async def read(self) -> Union[str, bytes]:
        """Read the resource content."""
        pass
