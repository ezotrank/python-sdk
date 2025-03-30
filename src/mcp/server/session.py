"""
ServerSession Module

This module provides the ServerSession class, which manages communication between the
server and client in the MCP (Model Context Protocol) framework. It is most commonly
used in MCP servers to interact with the client.

Common usage pattern:
```
    server = Server(name)

    @server.call_tool()
    async def handle_tool_call(ctx: RequestContext, arguments: dict[str, Any]) -> Any:
        # Check client capabilities before proceeding
        if ctx.session.check_client_capability(
            types.ClientCapabilities(experimental={"advanced_tools": dict()})
        ):
            # Perform advanced tool operations
            result = await perform_advanced_tool_operation(arguments)
        else:
            # Fall back to basic tool operations
            result = await perform_basic_tool_operation(arguments)

        return result

    @server.list_prompts()
    async def handle_list_prompts(ctx: RequestContext) -> list[types.Prompt]:
        # Access session for any necessary checks or operations
        if ctx.session.client_params:
            # Customize prompts based on client initialization parameters
            return generate_custom_prompts(ctx.session.client_params)
        else:
            return default_prompts
```

The ServerSession class is typically used internally by the Server class and should not
be instantiated directly by users of the MCP framework.
"""

from enum import Enum
from typing import Any, Optional, TypeVar, Union

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import AnyUrl

import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.shared.session import (
    BaseSession,
    RequestResponder,
)


class InitializationState(Enum):
    NotInitialized = 1
    Initializing = 2
    Initialized = 3


ServerSessionT = TypeVar("ServerSessionT", bound="ServerSession")

ServerRequestResponder = Union[
    RequestResponder[types.ClientRequest, types.ServerResult],
    types.ClientNotification,
    Exception,
]


class ServerSession(
    BaseSession[
        types.ServerRequest,
        types.ServerNotification,
        types.ServerResult,
        types.ClientRequest,
        types.ClientNotification,
    ]
):
    _initialized: InitializationState = InitializationState.NotInitialized
    _client_params: Optional[types.InitializeRequestParams] = None

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[Union[types.JSONRPCMessage, Exception]],
        write_stream: MemoryObjectSendStream[types.JSONRPCMessage],
        init_options: InitializationOptions,
    ) -> None:
        super().__init__(
            read_stream, write_stream, types.ClientRequest, types.ClientNotification
        )
        self._initialization_state = InitializationState.NotInitialized
        self._init_options = init_options
        self._incoming_message_stream_writer, self._incoming_message_stream_reader = (
            anyio.create_memory_object_stream[ServerRequestResponder](0)
        )
        self._exit_stack.push_async_callback(
            self._incoming_message_stream_reader.aclose
        )
        self._exit_stack.push_async_callback(
            self._incoming_message_stream_writer.aclose
        )

    @property
    def client_params(self) -> Optional[types.InitializeRequestParams]:
        return self._client_params

    def check_client_capability(self, capability: types.ClientCapabilities) -> bool:
        """Check if the client supports a specific capability."""
        if self._client_params is None:
            return False

        # Get client capabilities from initialization params
        client_caps = self._client_params.capabilities

        # Check each specified capability in the passed in capability object
        if capability.roots is not None:
            if client_caps.roots is None:
                return False
            if capability.roots.listChanged and not client_caps.roots.listChanged:
                return False

        if capability.sampling is not None:
            if client_caps.sampling is None:
                return False

        if capability.experimental is not None:
            if client_caps.experimental is None:
                return False
            # Check each experimental capability
            for exp_key, exp_value in capability.experimental.items():
                if (
                    exp_key not in client_caps.experimental
                    or client_caps.experimental[exp_key] != exp_value
                ):
                    return False

        return True

    async def _received_request(
        self, responder: RequestResponder[types.ClientRequest, types.ServerResult]
    ):
        request_root = responder.request.__root__
        if isinstance(request_root, types.InitializeRequest):
            params = request_root.params
            self._initialization_state = InitializationState.Initializing
            self._client_params = params
            with responder:
                await responder.respond(
                    types.ServerResult(
                        __root__=types.InitializeResult(
                            protocolVersion=types.LATEST_PROTOCOL_VERSION,
                            capabilities=self._init_options.capabilities,
                            serverInfo=types.Implementation(
                                name=self._init_options.server_name,
                                version=self._init_options.server_version,
                            ),
                            instructions=self._init_options.instructions,
                        )
                    )
                )
        else:
            if self._initialization_state != InitializationState.Initialized:
                raise RuntimeError("Received request before initialization was complete")
            # Forward other requests
            await self._handle_incoming(responder)


    async def _received_notification(
        self, notification: types.ClientNotification
    ) -> None:
        # Need this to avoid ASYNC910
        await anyio.lowlevel.checkpoint()
        notification_root = notification.__root__
        if isinstance(notification_root, types.InitializedNotification):
            self._initialization_state = InitializationState.Initialized
        else:
            if self._initialization_state != InitializationState.Initialized:
                raise RuntimeError(
                    "Received notification before initialization was complete"
                )
            # Forward other notifications
            await self._handle_incoming(notification)

    async def send_log_message(
        self, level: types.LoggingLevel, data: Any, logger: Optional[str] = None
    ) -> None:
        """Send a log message notification."""
        await self.send_notification(
            types.ServerNotification(
                __root__=types.LoggingMessageNotification(
                    method="notifications/message",
                    params=types.LoggingMessageNotificationParams(
                        level=level,
                        data=data,
                        logger=logger,
                    ),
                )
            )
        )

    async def send_resource_updated(self, uri: AnyUrl) -> None:
        """Send a resource updated notification."""
        await self.send_notification(
            types.ServerNotification(
                __root__=types.ResourceUpdatedNotification(
                    method="notifications/resources/updated",
                    params=types.ResourceUpdatedNotificationParams(uri=uri),
                )
            )
        )

    async def create_message(
        self,
        messages: list[types.SamplingMessage],
        *,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        include_context: Optional[types.IncludeContext] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        model_preferences: Optional[types.ModelPreferences] = None,
    ) -> types.CreateMessageResult:
        """Send a sampling/create_message request."""
        response_root = await self.send_request(
            types.ServerRequest(
                __root__=types.CreateMessageRequest(
                    method="sampling/createMessage",
                    params=types.CreateMessageRequestParams(
                        messages=messages,
                        systemPrompt=system_prompt,
                        includeContext=include_context,
                        temperature=temperature,
                        maxTokens=max_tokens,
                        stopSequences=stop_sequences,
                        metadata=metadata,
                        modelPreferences=model_preferences,
                    ),
                )
            ),
            types.CreateMessageResult,
        )
        # send_request returns the root model, we need the inner type
        if not isinstance(response_root.__root__, types.CreateMessageResult):
             raise TypeError(f"Expected CreateMessageResult, got {type(response_root.__root__)}")
        return response_root.__root__


    async def list_roots(self) -> types.ListRootsResult:
        """Send a roots/list request."""
        response_root = await self.send_request(
            types.ServerRequest(
                __root__=types.ListRootsRequest(
                    method="roots/list",
                )
            ),
            types.ListRootsResult,
        )
        # send_request returns the root model, we need the inner type
        if not isinstance(response_root.__root__, types.ListRootsResult):
             raise TypeError(f"Expected ListRootsResult, got {type(response_root.__root__)}")
        return response_root.__root__

    async def send_ping(self) -> types.EmptyResult:
        """Send a ping request."""
        response_root = await self.send_request(
            types.ServerRequest(
                __root__=types.PingRequest(
                    method="ping",
                )
            ),
            types.EmptyResult,
        )
        # send_request returns the root model, we need the inner type
        if not isinstance(response_root.__root__, types.EmptyResult):
             raise TypeError(f"Expected EmptyResult, got {type(response_root.__root__)}")
        return response_root.__root__

    async def send_progress_notification(
        self, progress_token: Union[str, int], progress: float, total: Optional[float] = None
    ) -> None:
        """Send a progress notification."""
        await self.send_notification(
            types.ServerNotification(
                __root__=types.ProgressNotification(
                    method="notifications/progress",
                    params=types.ProgressNotificationParams(
                        progressToken=progress_token,
                        progress=progress,
                        total=total,
                    ),
                )
            )
        )

    async def send_resource_list_changed(self) -> None:
        """Send a resource list changed notification."""
        await self.send_notification(
            types.ServerNotification(
                __root__=types.ResourceListChangedNotification(
                    method="notifications/resources/list_changed",
                )
            )
        )

    async def send_tool_list_changed(self) -> None:
        """Send a tool list changed notification."""
        await self.send_notification(
            types.ServerNotification(
                __root__=types.ToolListChangedNotification(
                    method="notifications/tools/list_changed",
                )
            )
        )

    async def send_prompt_list_changed(self) -> None:
        """Send a prompt list changed notification."""
        await self.send_notification(
            types.ServerNotification(
                __root__=types.PromptListChangedNotification(
                    method="notifications/prompts/list_changed",
                )
            )
        )

    async def _handle_incoming(self, req: ServerRequestResponder) -> None:
        await self._incoming_message_stream_writer.send(req)

    @property
    def incoming_messages(
        self,
    ) -> MemoryObjectReceiveStream[ServerRequestResponder]:
        return self._incoming_message_stream_reader
