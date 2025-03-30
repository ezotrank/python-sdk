import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import (
    # Annotated, # Not used by Pydantic V1 in this way
    Any,
    ForwardRef,
    Optional, # Use Optional for V1 compatibility
    Union, # Use Union for V1 compatibility
    Type, # For type hints
    Dict, # For type hints
    Tuple # For type hints
)

# Pydantic V1 imports
from pydantic import BaseModel, Field, create_model
# from pydantic import ConfigDict # V2 only
# from pydantic import WithJsonSchema # V2 only
# from pydantic._internal._typing_extra import eval_type_backport # V2 only
from pydantic.typing import evaluate_forwardref # V1 equivalent
# from pydantic.fields import FieldInfo # Less direct usage in V1 for this case
from pydantic.fields import Undefined # V1 undefined marker
# from pydantic_core import PydanticUndefined # V2 only

from mcp.server.fastmcp.exceptions import InvalidSignature
from mcp.server.fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


class ArgModelBase(BaseModel):
    """A model representing the arguments to a function."""

    def dict_one_level(self) -> Dict[str, Any]: # Renamed from model_dump_one_level
        """Return a dict of the model's fields, one level deep.

        That is, sub-models etc are not dumped - they are kept as pydantic models.
        """
        kwargs: Dict[str, Any] = {}
        for field_name in self.__fields__.keys(): # Use __fields__
            kwargs[field_name] = getattr(self, field_name)
        return kwargs

    # Use V1 Config class
    class Config:
        arbitrary_types_allowed = True


class FuncMetadata(BaseModel):
    # arg_model: Annotated[type[ArgModelBase], WithJsonSchema(None)] # V2 syntax
    arg_model: Type[ArgModelBase] # V1 syntax
    # We can add things in the future like
    #  - Maybe some args are excluded from attempting to parse from JSON
    #  - Maybe some args are special (like context) for dependency injection

    async def call_fn_with_arg_validation(
        self,
        fn: Union[Callable[..., Any], Awaitable[Any]], # V1 Union syntax
        fn_is_async: bool,
        arguments_to_validate: Dict[str, Any],
        arguments_to_pass_directly: Optional[Dict[str, Any]], # V1 Optional syntax
    ) -> Any:
        """Call the given function with arguments validated and injected.

        Arguments are first attempted to be parsed from JSON, then validated against
        the argument model, before being passed to the function.
        """
        arguments_pre_parsed = self.pre_parse_json(arguments_to_validate)
        # Use V1 parse_obj
        arguments_parsed_model = self.arg_model.parse_obj(arguments_pre_parsed)
        # Use renamed dict_one_level method
        arguments_parsed_dict = arguments_parsed_model.dict_one_level()

        # arguments_parsed_dict |= arguments_to_pass_directly or {} # Use update for Python < 3.9
        if arguments_to_pass_directly:
             arguments_parsed_dict.update(arguments_to_pass_directly)

        if fn_is_async:
             # Keep original isinstance check for pre-created awaitables
             if isinstance(fn, Awaitable):
                 # This case seems unlikely unless fn is e.g. a partial coroutine or Task
                 # Assuming it's a ready awaitable
                 return await fn
             # Check if the callable itself is an async function
             if inspect.iscoroutinefunction(fn):
                 return await fn(**arguments_parsed_dict)
             # If fn_is_async is True but fn is not awaitable/async func, await will raise TypeError
             # This matches the implicit behavior of the V2 code if fn was not awaitable.
             return await fn(**arguments_parsed_dict) # type: ignore

        # If not async, assume it's a regular callable
        if isinstance(fn, Callable):
             return fn(**arguments_parsed_dict)
        raise TypeError("fn must be either Callable or Awaitable")


    def pre_parse_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-parse data from JSON.

        Return a dict with same keys as input but with values parsed from JSON
        if appropriate.

        This is to handle cases like `["a", "b", "c"]` being passed in as JSON inside
        a string rather than an actual list. Claude desktop is prone to this - in fact
        it seems incapable of NOT doing this. For sub-models, it tends to pass
        dicts (JSON objects) as JSON strings, which can be pre-parsed here.
        """
        new_data = data.copy()  # Shallow copy
        # Use V1 __fields__
        for field_name, _field_info in self.arg_model.__fields__.items():
            if field_name not in data.keys():
                continue
            value_to_check = data[field_name]
            if isinstance(value_to_check, str):
                try:
                    # Attempt to parse string value as JSON
                    pre_parsed = json.loads(value_to_check)

                    # ONLY replace if the parsed result is a collection type (dict or list).
                    # If `value_to_check` was e.g. `"123"`, json.loads gives `123`.
                    # If `value_to_check` was `"\"hello\""`, json.loads gives `"hello"`.
                    # We want Pydantic V1 to handle these scalar/string cases from the original string.
                    if isinstance(pre_parsed, (dict, list)):
                         new_data[field_name] = pre_parsed
                    # else: keep original string for Pydantic V1 to parse

                except json.JSONDecodeError:
                    continue  # Not valid JSON, keep original string for Pydantic V1

        return new_data

    # Use V1 Config class
    class Config:
        arbitrary_types_allowed = True


def func_metadata(
    func: Callable[..., Any], skip_names: Sequence[str] = ()
) -> FuncMetadata:
    """Given a function, return metadata including a pydantic model representing its
    signature.

    The use case for this is
    ```
    meta = func_metadata(func) # Renamed func_to_pyd -> func_metadata
    validated_args = meta.arg_model.parse_obj(some_raw_data_dict) # Use parse_obj
    return func(**validated_args.dict_one_level()) # Use dict_one_level
    ```

    **critically** it also provides pre-parse helper to attempt to parse things from
    JSON.

    Args:
        func: The function to convert to a pydantic model
        skip_names: A list of parameter names to skip. These will not be included in
            the model.
    Returns:
        A pydantic model representing the function's signature.
    """
    sig = _get_typed_signature(func)
    params = sig.parameters
    dynamic_pydantic_model_params: Dict[str, Any] = {}
    globalns = getattr(func, "__globals__", {})
    # Try to get localns if it's a method (used for forward refs inside classes)
    localns = getattr(call, "__locals__", None) if hasattr(func, "__self__") else None


    for param in params.values():
        if param.name.startswith("_"):
            raise InvalidSignature(
                f"Parameter {param.name} of {func.__name__} cannot start with '_'"
            )
        if param.name in skip_names:
            continue

        annotation = param.annotation
        default_value = param.default

        # Resolve annotation, handling forward refs
        resolved_annotation = _get_typed_annotation(annotation, globalns, localns)

        # Determine Pydantic default
        pydantic_default = default_value if default_value is not inspect.Parameter.empty else Undefined

        # Handle untyped parameters (annotation is inspect.Parameter.empty)
        if resolved_annotation is inspect.Parameter.empty:
            resolved_annotation = Any

        # Ensure Optional type hint if default is None, unless already Any or Optional
        if pydantic_default is None and resolved_annotation is not Any:
             is_already_optional = False
             # Check if origin is Union and includes NoneType
             if hasattr(resolved_annotation, '__origin__') and resolved_annotation.__origin__ is Union:
                 if type(None) in getattr(resolved_annotation, '__args__', ()):
                     is_already_optional = True
             # Check if it's exactly NoneType (which implies optional)
             elif resolved_annotation is type(None):
                 is_already_optional = True

             if not is_already_optional:
                 resolved_annotation = Optional[resolved_annotation]

        # V1 create_model expects (type, default) or (type, FieldInfo)
        # We use Field to specify the default (which might be Undefined)
        field_definition = Field(pydantic_default)

        dynamic_pydantic_model_params[param.name] = (resolved_annotation, field_definition)

    # Create the model
    arguments_model = create_model(
        f"{func.__name__}Arguments",
        **dynamic_pydantic_model_params,
        __base__=ArgModelBase,
        # __module__ = func.__module__ # Helps with naming context
    )

    # Resolve any remaining forward references using the created model's context
    try:
        # Use model's update_forward_refs which handles local namespace better
        arguments_model.update_forward_refs(**globalns) # Pass globals primarily
    except Exception as e:
        # Log warning instead of raising, as some forward refs might be unresolvable
        # until runtime in specific contexts.
        logger.warning(f"Could not resolve forward refs for {func.__name__} model: {e}")

    resp = FuncMetadata(arg_model=arguments_model)
    return resp


# Updated signature to include localns
def _get_typed_annotation(annotation: Any, globalns: Dict[str, Any], localns: Optional[Dict[str, Any]]) -> Any:
    """Resolve annotation, evaluating ForwardRefs."""
    if isinstance(annotation, (str, ForwardRef)):
        # Ensure it's a ForwardRef instance
        fwd_ref = annotation if isinstance(annotation, ForwardRef) else ForwardRef(annotation)
        try:
            # evaluate_forwardref V1 signature: (type_, globalns, localns)
            # Fallback localns to globalns if not provided
            return evaluate_forwardref(fwd_ref, globalns, localns or globalns)
        except NameError as e:
            # Don't raise immediately, let update_forward_refs handle it later if possible.
            # Return the ForwardRef itself if evaluation fails here.
            logger.debug(f"Forward ref evaluation failed initially for {annotation}: {e}")
            return fwd_ref
        except Exception as e:
            # Raise for other unexpected errors during evaluation
            raise InvalidSignature(f"Error evaluating type annotation {annotation}: {e}") from e
    # Return non-string/ForwardRef annotations as is
    return annotation


def _get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    """Get function signature while evaluating forward references"""
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    # Try to get localns if it's a method (simplistic check)
    localns = getattr(call, "__locals__", None) if hasattr(call, "__self__") else None

    typed_params = []
    for param in signature.parameters.values():
         resolved_annotation = _get_typed_annotation(param.annotation, globalns, localns)
         typed_params.append(
             inspect.Parameter(
                 name=param.name,
                 kind=param.kind,
                 default=param.default,
                 annotation=resolved_annotation,
             )
         )

    # Try to resolve return annotation as well
    resolved_return_annotation = _get_typed_annotation(signature.return_annotation, globalns, localns)

    typed_signature = inspect.Signature(typed_params, return_annotation=resolved_return_annotation)
    return typed_signature
