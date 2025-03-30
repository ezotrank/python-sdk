"""
MCP Server Module

This module provides a framework for creating an MCP (Model Context Protocol) server.
It allows you to easily define and handle various types of requests and notifications
in an asynchronous manner.

Usage:
1. Create a Server instance:
   server = Server("your_server_name")

2. Define request handlers using decorators:
   @server.list_prompts()
   async def handle_list_prompts() -> list[types.Prompt]:
       # Implementation

   @server.get_prompt()
   async def handle_get_prompt(
       name: str, arguments: dict[str, str] | None
   ) -> types.GetPromptResult:
       # Implementation

   @server.list_tools()
   async def handle_list_tools() -> list[types.Tool]:
       # Implementation

   @server.call_tool()
   async def handle_call_tool(
       name: str, arguments: dict | None
   ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
       # Implementation

   @server.list_resource_templates()
   async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
       # Implementation

3. Define notification handlers if needed:
   @server.progress_notification()
   async def handle_progress(
       progress_token: str | int, progress: float, total: float | None
   ) -> None:
       # Implementation

4. Run the server:
   async def main():
       async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
           await server.run(
               read_stream,
               write_stream,
               InitializationOptions(
                   server_name="your_server_name",
                   server_version="your_version",
                   capabilities=server.get_capabilities(
                       notification_options=NotificationOptions(),
                       experimental_capabilities={},
                   ),
               ),
           )

   asyncio.run(main())

The Server class provides methods to register handlers for various MCP requests and
notifications. It automatically manages the request context and handles incoming
messages from the client.
"""

from __future__ import annotations as _annotations

import contextvars
import logging
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, Generic, TypeVar, Union

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import AnyUrl

import mcp.types as types
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server as stdio_server
from mcp.shared.context import RequestContext
from mcp.shared.exceptions import McpError
from mcp.shared.session import RequestResponder

logger = logging.getLogger(__name__)

LifespanResultT = TypeVar("LifespanResultT")

# This will be properly typed in each Server instance's context
request_ctx: contextvars.ContextVar[RequestContext[ServerSession, Any]] = (
    contextvars.ContextVar("request_ctx")
)


class NotificationOptions:
    def __init__(
        self,
        prompts_changed: bool = False,
        resources_changed: bool = False,
        tools_changed: bool = False,
    ):
        self.prompts_changed = prompts_changed
        self.resources_changed = resources_changed
        self.tools_changed = tools_changed


@asynccontextmanager
async def lifespan(server: Server[LifespanResultT]) -> AsyncIterator[object]:
    """Default lifespan context manager that does nothing.

    Args:
        server: The server instance this lifespan is managing

    Returns:
        An empty context object
    """
    yield {}


class Server(Generic[LifespanResultT]):
    def __init__(
        self,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
        lifespan: Callable[
            [Server[LifespanResultT]], AbstractAsyncContextManager[LifespanResultT]
        ] = lifespan,
    ):
        self.name = name
        self.version = version
        self.instructions = instructions
        self.lifespan = lifespan
        self.request_handlers: dict[
            type, Callable[..., Awaitable[types.ServerResult]]
        ] = {
            types.PingRequest: _ping_handler,
        }
        self.notification_handlers: dict[type, Callable[..., Awaitable[None]]] = {}
        self.notification_options = NotificationOptions()
        logger.debug(f"Initializing server '{name}'")

    def create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> InitializationOptions:
        """Create initialization options from this server instance."""

        def pkg_version(package: str) -> str:
            try:
                from importlib.metadata import version

                return version(package)
            except Exception:
                pass

            return "unknown"

        return InitializationOptions(
            server_name=self.name,
            server_version=self.version if self.version else pkg_version("mcp"),
            capabilities=self.get_capabilities(
                notification_options or NotificationOptions(),
                experimental_capabilities or {},
            ),
            instructions=self.instructions,
        )

    def get_capabilities(
        self,
        notification_options: NotificationOptions,
        experimental_capabilities: dict[str, dict[str, Any]],
    ) -> types.ServerCapabilities:
        """Convert existing handlers to a ServerCapabilities object."""
        prompts_capability = None
        resources_capability = None
        tools_capability = None
        logging_capability = None

        # Set prompt capabilities if handler exists
        if types.ListPromptsRequest in self.request_handlers:
            prompts_capability = types.PromptsCapability(
                listChanged=notification_options.prompts_changed
            )

        # Set resource capabilities if handler exists
        if types.ListResourcesRequest in self.request_handlers:
            resources_capability = types.ResourcesCapability(
                subscribe=False, listChanged=notification_options.resources_changed
            )

        # Set tool capabilities if handler exists
        if types.ListToolsRequest in self.request_handlers:
            tools_capability = types.ToolsCapability(
                listChanged=notification_options.tools_changed
            )

        # Set logging capabilities if handler exists
        if types.SetLevelRequest in self.request_handlers:
            logging_capability = types.LoggingCapability()

        return types.ServerCapabilities(
            prompts=prompts_capability,
            resources=resources_capability,
            tools=tools_capability,
            logging=logging_capability,
            experimental=experimental_capabilities,
        )

    @property
    def request_context(self) -> RequestContext[ServerSession, LifespanResultT]:
        """If called outside of a request context, this will raise a LookupError."""
        return request_ctx.get()

    def list_prompts(self):
        def decorator(func: Callable[[], Awaitable[list[types.Prompt]]]):
            logger.debug("Registering handler for PromptListRequest")

            async def handler(_: Any):
                prompts = await func()
                return types.ServerResult(__root__=types.ListPromptsResult(prompts=prompts))

            self.request_handlers[types.ListPromptsRequest] = handler
            return func

        return decorator

    def get_prompt(self):
        def decorator(
            func: Callable[
                [str, dict[str, str] | None], Awaitable[types.GetPromptResult]
            ],
        ):
            logger.debug("Registering handler for GetPromptRequest")

            async def handler(req: types.GetPromptRequest):
                prompt_get = await func(req.params.name, req.params.arguments)
                return types.ServerResult(__root__=prompt_get)

            self.request_handlers[types.GetPromptRequest] = handler
            return func

        return decorator

    def list_resources(self):
        def decorator(func: Callable[[], Awaitable[list[types.Resource]]]):
            logger.debug("Registering handler for ListResourcesRequest")

            async def handler(_: Any):
                resources = await func()
                return types.ServerResult(
                    __root__=types.ListResourcesResult(resources=resources)
                )

            self.request_handlers[types.ListResourcesRequest] = handler
            return func

        return decorator

    def list_resource_templates(self):
        def decorator(func: Callable[[], Awaitable[list[types.ResourceTemplate]]]):
            logger.debug("Registering handler for ListResourceTemplatesRequest")

            async def handler(_: Any):
                templates = await func()
                return types.ServerResult(
                    __root__=types.ListResourceTemplatesResult(resourceTemplates=templates)
                )

            self.request_handlers[types.ListResourceTemplatesRequest] = handler
            return func

        return decorator

    def read_resource(self):
        def decorator(
            func: Callable[
                [AnyUrl], Awaitable[Union[str, bytes, Iterable[ReadResourceContents]]]
            ],
        ):
            logger.debug("Registering handler for ReadResourceRequest")

            async def handler(req: types.ReadResourceRequest):
                result = await func(req.params.uri)

                def create_content(data: Union[str, bytes], mime_type: str | None):
                    if isinstance(data, str):
                        return types.TextResourceContents(
                            uri=req.params.uri,
                            text=data,
                            mimeType=mime_type or "text/plain",
                        )
                    elif isinstance(data, bytes):
                        import base64
                        return types.BlobResourceContents(
                            uri=req.params.uri,
                            blob=base64.b64encode(data).decode(),
                            mimeType=mime_type or "application/octet-stream",
                        )
                    else:
                        # This case should technically not be reachable due to type hints
                        # but added for robustness.
                        raise TypeError(f"Unsupported data type: {type(data)}")


                if isinstance(result, (str, bytes)):
                     warnings.warn(
                         "Returning str or bytes from read_resource is deprecated. "
                         "Use Iterable[ReadResourceContents] instead.",
                         DeprecationWarning,
                         stacklevel=2,
                     )
                     content = create_content(result, None)
                     return types.ServerResult(
                         __root__=types.ReadResourceResult(
                             contents=[content],
                         )
                     )
                elif isinstance(result, Iterable):
                     contents_list = [
                         create_content(content_item.content, content_item.mime_type)
                         for content_item in result # Corrected from `contents` to `result`
                     ]
                     return types.ServerResult(
                         __root__=types.ReadResourceResult(
                             contents=contents_list,
                         )
                     )
                else:
                    raise ValueError(
                        f"Unexpected return type from read_resource: {type(result)}"
                    )

            self.request_handlers[types.ReadResourceRequest] = handler
            return func

        return decorator

    def set_logging_level(self):
        def decorator(func: Callable[[types.LoggingLevel], Awaitable[None]]):
            logger.debug("Registering handler for SetLevelRequest")

            async def handler(req: types.SetLevelRequest):
                await func(req.params.level)
                return types.ServerResult(__root__=types.EmptyResult())

            self.request_handlers[types.SetLevelRequest] = handler
            return func

        return decorator

    def subscribe_resource(self):
        def decorator(func: Callable[[AnyUrl], Awaitable[None]]):
            logger.debug("Registering handler for SubscribeRequest")

            async def handler(req: types.SubscribeRequest):
                await func(req.params.uri)
                return types.ServerResult(__root__=types.EmptyResult())

            self.request_handlers[types.SubscribeRequest] = handler
            return func

        return decorator

    def unsubscribe_resource(self):
        def decorator(func: Callable[[AnyUrl], Awaitable[None]]):
            logger.debug("Registering handler for UnsubscribeRequest")

            async def handler(req: types.UnsubscribeRequest):
                await func(req.params.uri)
                return types.ServerResult(__root__=types.EmptyResult())

            self.request_handlers[types.UnsubscribeRequest] = handler
            return func

        return decorator

    def list_tools(self):
        def decorator(func: Callable[[], Awaitable[list[types.Tool]]]):
            logger.debug("Registering handler for ListToolsRequest")

            async def handler(_: Any):
                tools = await func()
                return types.ServerResult(__root__=types.ListToolsResult(tools=tools))

            self.request_handlers[types.ListToolsRequest] = handler
            return func

        return decorator

    def call_tool(self):
        def decorator(
            func: Callable[
                ...,
                Awaitable[
                    Iterable[
                        Union[types.TextContent, types.ImageContent, types.EmbeddedResource]
                    ]
                ],
            ],
        ):
            logger.debug("Registering handler for CallToolRequest")

            async def handler(req: types.CallToolRequest):
                try:
                    results = await func(req.params.name, (req.params.arguments or {}))
                    return types.ServerResult(
                        __root__=types.CallToolResult(content=list(results), isError=False)
                    )
                except Exception as e:
                    return types.ServerResult(
                        __root__=types.CallToolResult(
                            content=[types.TextContent(type="text", text=str(e))],
                            isError=True,
                        )
                    )

            self.request_handlers[types.CallToolRequest] = handler
            return func

        return decorator

    def progress_notification(self):
        def decorator(
            func: Callable[[Union[str, int], float, float | None], Awaitable[None]],
        ):
            logger.debug("Registering handler for ProgressNotification")

            async def handler(req: types.ProgressNotification):
                await func(
                    req.params.progressToken, req.params.progress, req.params.total
                )

            self.notification_handlers[types.ProgressNotification] = handler
            return func

        return decorator

    def completion(self):
        """Provides completions for prompts and resource templates"""

        def decorator(
            func: Callable[
                [
                    Union[types.PromptReference, types.ResourceReference],
                    types.CompletionArgument,
                ],
                Awaitable[types.Completion | None],
            ],
        ):
            logger.debug("Registering handler for CompleteRequest")

            async def handler(req: types.CompleteRequest):
                completion = await func(req.params.ref, req.params.argument)
                return types.ServerResult(
                    __root__=types.CompleteResult(
                        completion=completion
                        if completion is not None
                        else types.Completion(values=[], total=None, hasMore=None),
                    )
                )

            self.request_handlers[types.CompleteRequest] = handler
            return func

        return decorator

    async def run(
        self,
        read_stream: MemoryObjectReceiveStream[Union[types.JSONRPCMessage, Exception]],
        write_stream: MemoryObjectSendStream[types.JSONRPCMessage],
        initialization_options: InitializationOptions,
        # When False, exceptions are returned as messages to the client.
        # When True, exceptions are raised, which will cause the server to shut down
        # but also make tracing exceptions much easier during testing and when using
        # in-process servers.
        raise_exceptions: bool = False,
    ):
        async with AsyncExitStack() as stack:
            lifespan_context = await stack.enter_async_context(self.lifespan(self))
            session = await stack.enter_async_context(
                ServerSession(read_stream, write_stream, initialization_options)
            )

            async with anyio.create_task_group() as tg:
                async for message_union in session.incoming_messages:
                    logger.debug(f"Received message: {message_union}")

                    # Extract the actual message or exception
                    if isinstance(message_union, Exception):
                         message_obj = message_union
                    elif isinstance(message_union, types.JSONRPCMessage):
                         message_obj = message_union.__root__ # Get underlying obj
                    else:
                         # Handle RequestResponder which wraps the request
                         message_obj = message_union

                    tg.start_soon(
                        self._handle_message,
                        message_obj, # Pass the extracted object or exception
                        session,
                        lifespan_context,
                        raise_exceptions,
                    )

    async def _handle_message(
        self,
        message: Union[
             RequestResponder[types.ClientRequest, types.ServerResult],
             types.JSONRPCNotification, # Use specific notification type
             Exception
        ],
        session: ServerSession,
        lifespan_context: LifespanResultT,
        raise_exceptions: bool = False,
    ):
        with warnings.catch_warnings(record=True) as w:
            if isinstance(message, RequestResponder):
                 # We know message.request is ClientRequest which has __root__
                 req = message.request.__root__
                 with message: # Use the responder context manager
                      await self._handle_request(
                           message, req, session, lifespan_context, raise_exceptions
                      )
            elif isinstance(message, types.JSONRPCNotification):
                 # message itself is the notification object from the union
                 await self._handle_notification(message)
            elif isinstance(message, Exception):
                 # Handle exceptions if needed, e.g., log them
                 logger.error(f"Received exception directly: {message}")
                 if raise_exceptions:
                      raise message
            else:
                 logger.warning(f"Unhandled message type: {type(message)}")


            for warning in w:
                logger.info(f"Warning: {warning.category.__name__}: {warning.message}")

    async def _handle_request(
        self,
        # Keep original responder for responding
        responder: RequestResponder[types.ClientRequest, types.ServerResult],
        # Pass the actual request object (extracted from __root__)
        req: Union[
                types.PingRequest, types.InitializeRequest, types.CompleteRequest,
                types.SetLevelRequest, types.GetPromptRequest, types.ListPromptsRequest,
                types.ListResourcesRequest, types.ListResourceTemplatesRequest,
                types.ReadResourceRequest, types.SubscribeRequest, types.UnsubscribeRequest,
                types.CallToolRequest, types.ListToolsRequest
            ],
        session: ServerSession,
        lifespan_context: LifespanResultT,
        raise_exceptions: bool,
    ):
        logger.info(f"Processing request of type {type(req).__name__}")
        req_type = type(req)
        if req_type in self.request_handlers:
            handler = self.request_handlers[req_type]
            logger.debug(f"Dispatching request of type {req_type.__name__}")

            token = None
            try:
                # Set our global state that can be retrieved via
                # app.get_request_context()
                token = request_ctx.set(
                    RequestContext(
                        responder.request_id,
                        responder.request_meta,
                        session,
                        lifespan_context,
                    )
                )
                response_result = await handler(req) # handler returns ServerResult
                response_data = response_result.__root__ # Extract data from __root__

            except McpError as err:
                # McpError already contains the ErrorData payload
                response_data = err.error
            except Exception as err:
                logger.exception("Error handling request") # Log with traceback
                if raise_exceptions:
                    raise err
                # Create ErrorData payload
                response_data = types.ErrorData(code=types.INTERNAL_ERROR, message=str(err), data=None)
            finally:
                # Reset the global state after we are done
                if token is not None:
                    request_ctx.reset(token)

            # Respond using the original responder
            await responder.respond(response_data)
        else:
            # Respond using the original responder
            await responder.respond(
                types.ErrorData(
                    code=types.METHOD_NOT_FOUND,
                    message=f"Method not found: {getattr(req, 'method', type(req).__name__)}",
                )
            )

        logger.debug(f"Response sent for request ID {responder.request_id}")

    async def _handle_notification(self, notify: types.JSONRPCNotification):
        # Extract the actual notification object from the __root__ of ClientNotification if needed
        # In this version, _handle_message passes the JSONRPCNotification directly
        actual_notify_obj = notify # Already the correct type
        notify_type = type(actual_notify_obj)

        if notify_type in self.notification_handlers:
            handler = self.notification_handlers[notify_type]
            logger.debug(
                f"Dispatching notification of type {notify_type.__name__}"
            )

            try:
                # Pass the actual notification object to the handler
                await handler(actual_notify_obj)
            except Exception as err:
                logger.error(f"Uncaught exception in notification handler: {err}", exc_info=True)
        else:
            logger.warning(f"No handler registered for notification type {notify_type.__name__}")


async def _ping_handler(request: types.PingRequest) -> types.ServerResult:
    # For Pydantic v1 RootModel, wrap the result
    return types.ServerResult(__root__=types.EmptyResult())
