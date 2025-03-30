from collections.abc import Callable
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Extra, Field, FileUrl
from pydantic.networks import AnyUrl

"""
Model Context Protocol bindings for Python

These bindings were generated from https://github.com/modelcontextprotocol/specification,
using Claude, with a prompt something like the following:

Generate idiomatic Python bindings for this schema for MCP, or the "Model Context
Protocol." The schema is defined in TypeScript, but there's also a JSON Schema version
for reference.

* For the bindings, let's use Pydantic V2 models. --> Use Pydantic V1.10.17
* Each model should allow extra fields everywhere, by specifying `class Config: extra = Extra.allow`. Do this in every case, instead of a custom base class.
* Union types should be represented with a Pydantic `__root__: Union[...]`.
* Define additional model classes instead of using dictionaries. Do this even if they're
  not separate types in the schema.
"""

LATEST_PROTOCOL_VERSION = "2024-11-05"

ProgressToken = Union[str, int]
Cursor = str
Role = Literal["user", "assistant"]
RequestId = Union[str, int]
AnyFunction: TypeAlias = Callable[..., Any]


class RequestParams(BaseModel):
    class Meta(BaseModel):
        progressToken: Optional[ProgressToken] = None
        """
        If specified, the caller requests out-of-band progress notifications for
        this request (as represented by notifications/progress). The value of this
        parameter is an opaque token that will be attached to any subsequent
        notifications. The receiver is not obligated to provide these notifications.
        """

        class Config:
            extra = Extra.allow

    meta: Optional[Meta] = Field(alias="_meta", default=None)

    class Config:
        extra = Extra.allow


class NotificationParams(BaseModel):
    class Meta(BaseModel):
        class Config:
            extra = Extra.allow

    meta: Optional[Meta] = Field(alias="_meta", default=None)
    """
    This parameter name is reserved by MCP to allow clients and servers to attach
    additional metadata to their notifications.
    """
    class Config:
        extra = Extra.allow


RequestParamsT = TypeVar("RequestParamsT", bound=Union[RequestParams, dict[str, Any], None])
NotificationParamsT = TypeVar(
    "NotificationParamsT", bound=Union[NotificationParams, dict[str, Any], None]
)
MethodT = TypeVar("MethodT", bound=str)


class Request(BaseModel, Generic[RequestParamsT, MethodT]):
    """Base class for JSON-RPC requests."""

    method: MethodT
    params: RequestParamsT

    class Config:
        extra = Extra.allow


class PaginatedRequest(Request[RequestParamsT, MethodT]):
    cursor: Optional[Cursor] = None
    """
    An opaque token representing the current pagination position.
    If provided, the server should return results starting after this cursor.
    """


class Notification(BaseModel, Generic[NotificationParamsT, MethodT]):
    """Base class for JSON-RPC notifications."""

    method: MethodT
    params: NotificationParamsT

    class Config:
        extra = Extra.allow


class Result(BaseModel):
    """Base class for JSON-RPC results."""

    meta: Optional[dict[str, Any]] = Field(alias="_meta", default=None)
    """
    This result property is reserved by the protocol to allow clients and servers to
    attach additional metadata to their responses.
    """
    class Config:
        extra = Extra.allow


class PaginatedResult(Result):
    nextCursor: Optional[Cursor] = None
    """
    An opaque token representing the pagination position after the last returned result.
    If present, there may be more results available.
    """


class JSONRPCRequest(Request[Optional[dict[str, Any]], str]):
    """A request that expects a response."""

    jsonrpc: Literal["2.0"]
    id: RequestId
    method: str
    params: Optional[dict[str, Any]] = None


class JSONRPCNotification(Notification[Optional[dict[str, Any]], str]):
    """A notification which does not expect a response."""

    jsonrpc: Literal["2.0"]
    params: Optional[dict[str, Any]] = None
    method: str # Required by base Notification class


class JSONRPCResponse(BaseModel):
    """A successful (non-error) response to a request."""

    jsonrpc: Literal["2.0"]
    id: RequestId
    result: dict[str, Any]

    class Config:
        extra = Extra.allow


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ErrorData(BaseModel):
    """Error information for JSON-RPC error responses."""

    code: int
    """The error type that occurred."""

    message: str
    """
    A short description of the error. The message SHOULD be limited to a concise single
    sentence.
    """

    data: Optional[Any] = None
    """
    Additional information about the error. The value of this member is defined by the
    sender (e.g. detailed error information, nested errors etc.).
    """

    class Config:
        extra = Extra.allow


class JSONRPCError(BaseModel):
    """A response to a request that indicates an error occurred."""

    jsonrpc: Literal["2.0"]
    id: Union[str, int]
    error: ErrorData

    class Config:
        extra = Extra.allow


class JSONRPCMessage(BaseModel):
    __root__: Union[JSONRPCRequest, JSONRPCNotification, JSONRPCResponse, JSONRPCError]


class EmptyResult(Result):
    """A response that indicates success but carries no data."""


class Implementation(BaseModel):
    """Describes the name and version of an MCP implementation."""

    name: str
    version: str

    class Config:
        extra = Extra.allow


class RootsCapability(BaseModel):
    """Capability for root operations."""

    listChanged: Optional[bool] = None
    """Whether the client supports notifications for changes to the roots list."""

    class Config:
        extra = Extra.allow


class SamplingCapability(BaseModel):
    """Capability for logging operations."""

    class Config:
        extra = Extra.allow


class ClientCapabilities(BaseModel):
    """Capabilities a client may support."""

    experimental: Optional[dict[str, dict[str, Any]]] = None
    """Experimental, non-standard capabilities that the client supports."""
    sampling: Optional[SamplingCapability] = None
    """Present if the client supports sampling from an LLM."""
    roots: Optional[RootsCapability] = None
    """Present if the client supports listing roots."""

    class Config:
        extra = Extra.allow


class PromptsCapability(BaseModel):
    """Capability for prompts operations."""

    listChanged: Optional[bool] = None
    """Whether this server supports notifications for changes to the prompt list."""

    class Config:
        extra = Extra.allow


class ResourcesCapability(BaseModel):
    """Capability for resources operations."""

    subscribe: Optional[bool] = None
    """Whether this server supports subscribing to resource updates."""
    listChanged: Optional[bool] = None
    """Whether this server supports notifications for changes to the resource list."""

    class Config:
        extra = Extra.allow


class ToolsCapability(BaseModel):
    """Capability for tools operations."""

    listChanged: Optional[bool] = None
    """Whether this server supports notifications for changes to the tool list."""

    class Config:
        extra = Extra.allow


class LoggingCapability(BaseModel):
    """Capability for logging operations."""

    class Config:
        extra = Extra.allow


class ServerCapabilities(BaseModel):
    """Capabilities that a server may support."""

    experimental: Optional[dict[str, dict[str, Any]]] = None
    """Experimental, non-standard capabilities that the server supports."""
    logging: Optional[LoggingCapability] = None
    """Present if the server supports sending log messages to the client."""
    prompts: Optional[PromptsCapability] = None
    """Present if the server offers any prompt templates."""
    resources: Optional[ResourcesCapability] = None
    """Present if the server offers any resources to read."""
    tools: Optional[ToolsCapability] = None
    """Present if the server offers any tools to call."""

    class Config:
        extra = Extra.allow


class InitializeRequestParams(RequestParams):
    """Parameters for the initialize request."""

    protocolVersion: Union[str, int]
    """The latest version of the Model Context Protocol that the client supports."""
    capabilities: ClientCapabilities
    clientInfo: Implementation

    class Config:
        extra = Extra.allow


class InitializeRequest(Request[InitializeRequestParams, Literal["initialize"]]):
    """
    This request is sent from the client to the server when it first connects, asking it
    to begin initialization.
    """

    method: Literal["initialize"] = "initialize"
    params: InitializeRequestParams


class InitializeResult(Result):
    """After receiving an initialize request from the client, the server sends this."""

    protocolVersion: Union[str, int]
    """The version of the Model Context Protocol that the server wants to use."""
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: Optional[str] = None
    """Instructions describing how to use the server and its features."""


class InitializedNotification(
    Notification[Optional[NotificationParams], Literal["notifications/initialized"]]
):
    """
    This notification is sent from the client to the server after initialization has
    finished.
    """

    method: Literal["notifications/initialized"] = "notifications/initialized"
    params: Optional[NotificationParams] = None


class PingRequest(Request[Optional[RequestParams], Literal["ping"]]):
    """
    A ping, issued by either the server or the client, to check that the other party is
    still alive.
    """

    method: Literal["ping"] = "ping"
    params: Optional[RequestParams] = None


class ProgressNotificationParams(NotificationParams):
    """Parameters for progress notifications."""

    progressToken: ProgressToken
    """
    The progress token which was given in the initial request, used to associate this
    notification with the request that is proceeding.
    """
    progress: float
    """
    The progress thus far. This should increase every time progress is made, even if the
    total is unknown.
    """
    total: Optional[float] = None
    """Total number of items to process (or total progress required), if known."""

    class Config:
        extra = Extra.allow


class ProgressNotification(
    Notification[ProgressNotificationParams, Literal["notifications/progress"]]
):
    """
    An out-of-band notification used to inform the receiver of a progress update for a
    long-running request.
    """

    method: Literal["notifications/progress"] = "notifications/progress"
    params: ProgressNotificationParams


class ListResourcesRequest(
    PaginatedRequest[Optional[RequestParams], Literal["resources/list"]]
):
    """Sent from the client to request a list of resources the server has."""

    method: Literal["resources/list"] = "resources/list"
    params: Optional[RequestParams] = None


class Annotations(BaseModel):
    audience: Optional[list[Role]] = None
    priority: Optional[float] = Field(None, ge=0.0, le=1.0)

    class Config:
        extra = Extra.allow


class Resource(BaseModel):
    """A known resource that the server is capable of reading."""

    uri: AnyUrl
    """The URI of this resource."""
    name: str
    """A human-readable name for this resource."""
    description: Optional[str] = None
    """A description of what this resource represents."""
    mimeType: Optional[str] = None
    """The MIME type of this resource, if known."""
    size: Optional[int] = None
    """
    The size of the raw resource content, in bytes (i.e., before base64 encoding
    or any tokenization), if known.

    This can be used by Hosts to display file sizes and estimate context window usage.
    """
    annotations: Optional[Annotations] = None

    class Config:
        extra = Extra.allow


class ResourceTemplate(BaseModel):
    """A template description for resources available on the server."""

    uriTemplate: str
    """
    A URI template (according to RFC 6570) that can be used to construct resource
    URIs.
    """
    name: str
    """A human-readable name for the type of resource this template refers to."""
    description: Optional[str] = None
    """A human-readable description of what this template is for."""
    mimeType: Optional[str] = None
    """
    The MIME type for all resources that match this template. This should only be
    included if all resources matching this template have the same type.
    """
    annotations: Optional[Annotations] = None

    class Config:
        extra = Extra.allow


class ListResourcesResult(PaginatedResult):
    """The server's response to a resources/list request from the client."""

    resources: list[Resource]


class ListResourceTemplatesRequest(
    PaginatedRequest[Optional[RequestParams], Literal["resources/templates/list"]]
):
    """Sent from the client to request a list of resource templates the server has."""

    method: Literal["resources/templates/list"] = "resources/templates/list"
    params: Optional[RequestParams] = None


class ListResourceTemplatesResult(PaginatedResult):
    """The server's response to a resources/templates/list request from the client."""

    resourceTemplates: list[ResourceTemplate]


class ReadResourceRequestParams(RequestParams):
    """Parameters for reading a resource."""

    uri: AnyUrl
    """
    The URI of the resource to read. The URI can use any protocol; it is up to the
    server how to interpret it.
    """

    class Config:
        extra = Extra.allow


class ReadResourceRequest(
    Request[ReadResourceRequestParams, Literal["resources/read"]]
):
    """Sent from the client to the server, to read a specific resource URI."""

    method: Literal["resources/read"] = "resources/read"
    params: ReadResourceRequestParams


class ResourceContents(BaseModel):
    """The contents of a specific resource or sub-resource."""

    uri: AnyUrl
    """The URI of this resource."""
    mimeType: Optional[str] = None
    """The MIME type of this resource, if known."""

    class Config:
        extra = Extra.allow


class TextResourceContents(ResourceContents):
    """Text contents of a resource."""

    text: str
    """
    The text of the item. This must only be set if the item can actually be represented
    as text (not binary data).
    """


class BlobResourceContents(ResourceContents):
    """Binary contents of a resource."""

    blob: str
    """A base64-encoded string representing the binary data of the item."""


class ReadResourceResult(Result):
    """The server's response to a resources/read request from the client."""

    contents: list[Union[TextResourceContents, BlobResourceContents]]


class ResourceListChangedNotification(
    Notification[
        Optional[NotificationParams], Literal["notifications/resources/list_changed"]
    ]
):
    """
    An optional notification from the server to the client, informing it that the list
    of resources it can read from has changed.
    """

    method: Literal["notifications/resources/list_changed"] = "notifications/resources/list_changed"
    params: Optional[NotificationParams] = None


class SubscribeRequestParams(RequestParams):
    """Parameters for subscribing to a resource."""

    uri: AnyUrl
    """
    The URI of the resource to subscribe to. The URI can use any protocol; it is up to
    the server how to interpret it.
    """

    class Config:
        extra = Extra.allow


class SubscribeRequest(Request[SubscribeRequestParams, Literal["resources/subscribe"]]):
    """
    Sent from the client to request resources/updated notifications from the server
    whenever a particular resource changes.
    """

    method: Literal["resources/subscribe"] = "resources/subscribe"
    params: SubscribeRequestParams


class UnsubscribeRequestParams(RequestParams):
    """Parameters for unsubscribing from a resource."""

    uri: AnyUrl
    """The URI of the resource to unsubscribe from."""

    class Config:
        extra = Extra.allow


class UnsubscribeRequest(
    Request[UnsubscribeRequestParams, Literal["resources/unsubscribe"]]
):
    """
    Sent from the client to request cancellation of resources/updated notifications from
    the server.
    """

    method: Literal["resources/unsubscribe"] = "resources/unsubscribe"
    params: UnsubscribeRequestParams


class ResourceUpdatedNotificationParams(NotificationParams):
    """Parameters for resource update notifications."""

    uri: AnyUrl
    """
    The URI of the resource that has been updated. This might be a sub-resource of the
    one that the client actually subscribed to.
    """

    class Config:
        extra = Extra.allow


class ResourceUpdatedNotification(
    Notification[
        ResourceUpdatedNotificationParams, Literal["notifications/resources/updated"]
    ]
):
    """
    A notification from the server to the client, informing it that a resource has
    changed and may need to be read again.
    """

    method: Literal["notifications/resources/updated"] = "notifications/resources/updated"
    params: ResourceUpdatedNotificationParams


class ListPromptsRequest(
    PaginatedRequest[Optional[RequestParams], Literal["prompts/list"]]
):
    """Sent from the client to request a list of prompts and prompt templates."""

    method: Literal["prompts/list"] = "prompts/list"
    params: Optional[RequestParams] = None


class PromptArgument(BaseModel):
    """An argument for a prompt template."""

    name: str
    """The name of the argument."""
    description: Optional[str] = None
    """A human-readable description of the argument."""
    required: Optional[bool] = None
    """Whether this argument must be provided."""

    class Config:
        extra = Extra.allow


class Prompt(BaseModel):
    """A prompt or prompt template that the server offers."""

    name: str
    """The name of the prompt or prompt template."""
    description: Optional[str] = None
    """An optional description of what this prompt provides."""
    arguments: Optional[list[PromptArgument]] = None
    """A list of arguments to use for templating the prompt."""

    class Config:
        extra = Extra.allow


class ListPromptsResult(PaginatedResult):
    """The server's response to a prompts/list request from the client."""

    prompts: list[Prompt]


class GetPromptRequestParams(RequestParams):
    """Parameters for getting a prompt."""

    name: str
    """The name of the prompt or prompt template."""
    arguments: Optional[dict[str, str]] = None
    """Arguments to use for templating the prompt."""

    class Config:
        extra = Extra.allow


class GetPromptRequest(Request[GetPromptRequestParams, Literal["prompts/get"]]):
    """Used by the client to get a prompt provided by the server."""

    method: Literal["prompts/get"] = "prompts/get"
    params: GetPromptRequestParams


class TextContent(BaseModel):
    """Text content for a message."""

    type: Literal["text"]
    text: str
    """The text content of the message."""
    annotations: Optional[Annotations] = None

    class Config:
        extra = Extra.allow


class ImageContent(BaseModel):
    """Image content for a message."""

    type: Literal["image"]
    data: str
    """The base64-encoded image data."""
    mimeType: str
    """
    The MIME type of the image. Different providers may support different
    image types.
    """
    annotations: Optional[Annotations] = None

    class Config:
        extra = Extra.allow


class SamplingMessage(BaseModel):
    """Describes a message issued to or received from an LLM API."""

    role: Role
    content: Union[TextContent, ImageContent]

    class Config:
        extra = Extra.allow


class EmbeddedResource(BaseModel):
    """
    The contents of a resource, embedded into a prompt or tool call result.

    It is up to the client how best to render embedded resources for the benefit
    of the LLM and/or the user.
    """

    type: Literal["resource"]
    resource: Union[TextResourceContents, BlobResourceContents]
    annotations: Optional[Annotations] = None

    class Config:
        extra = Extra.allow


class PromptMessage(BaseModel):
    """Describes a message returned as part of a prompt."""

    role: Role
    content: Union[TextContent, ImageContent, EmbeddedResource]

    class Config:
        extra = Extra.allow


class GetPromptResult(Result):
    """The server's response to a prompts/get request from the client."""

    description: Optional[str] = None
    """An optional description for the prompt."""
    messages: list[PromptMessage]


class PromptListChangedNotification(
    Notification[
        Optional[NotificationParams], Literal["notifications/prompts/list_changed"]
    ]
):
    """
    An optional notification from the server to the client, informing it that the list
    of prompts it offers has changed.
    """

    method: Literal["notifications/prompts/list_changed"] = "notifications/prompts/list_changed"
    params: Optional[NotificationParams] = None


class ListToolsRequest(PaginatedRequest[Optional[RequestParams], Literal["tools/list"]]):
    """Sent from the client to request a list of tools the server has."""

    method: Literal["tools/list"] = "tools/list"
    params: Optional[RequestParams] = None


class Tool(BaseModel):
    """Definition for a tool the client can call."""

    name: str
    """The name of the tool."""
    description: Optional[str] = None
    """A human-readable description of the tool."""
    inputSchema: dict[str, Any]
    """A JSON Schema object defining the expected parameters for the tool."""

    class Config:
        extra = Extra.allow


class ListToolsResult(PaginatedResult):
    """The server's response to a tools/list request from the client."""

    tools: list[Tool]


class CallToolRequestParams(RequestParams):
    """Parameters for calling a tool."""

    name: str
    arguments: Optional[dict[str, Any]] = None

    class Config:
        extra = Extra.allow


class CallToolRequest(Request[CallToolRequestParams, Literal["tools/call"]]):
    """Used by the client to invoke a tool provided by the server."""

    method: Literal["tools/call"] = "tools/call"
    params: CallToolRequestParams


class CallToolResult(Result):
    """The server's response to a tool call."""

    content: list[Union[TextContent, ImageContent, EmbeddedResource]]
    isError: bool = False


class ToolListChangedNotification(
    Notification[Optional[NotificationParams], Literal["notifications/tools/list_changed"]]
):
    """
    An optional notification from the server to the client, informing it that the list
    of tools it offers has changed.
    """

    method: Literal["notifications/tools/list_changed"] = "notifications/tools/list_changed"
    params: Optional[NotificationParams] = None


LoggingLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"
]


class SetLevelRequestParams(RequestParams):
    """Parameters for setting the logging level."""

    level: LoggingLevel
    """The level of logging that the client wants to receive from the server."""

    class Config:
        extra = Extra.allow


class SetLevelRequest(Request[SetLevelRequestParams, Literal["logging/setLevel"]]):
    """A request from the client to the server, to enable or adjust logging."""

    method: Literal["logging/setLevel"] = "logging/setLevel"
    params: SetLevelRequestParams


class LoggingMessageNotificationParams(NotificationParams):
    """Parameters for logging message notifications."""

    level: LoggingLevel
    """The severity of this log message."""
    logger: Optional[str] = None
    """An optional name of the logger issuing this message."""
    data: Any
    """
    The data to be logged, such as a string message or an object. Any JSON serializable
    type is allowed here.
    """

    class Config:
        extra = Extra.allow


class LoggingMessageNotification(
    Notification[LoggingMessageNotificationParams, Literal["notifications/message"]]
):
    """Notification of a log message passed from server to client."""

    method: Literal["notifications/message"] = "notifications/message"
    params: LoggingMessageNotificationParams


IncludeContext = Literal["none", "thisServer", "allServers"]


class ModelHint(BaseModel):
    """Hints to use for model selection."""

    name: Optional[str] = None
    """A hint for a model name."""

    class Config:
        extra = Extra.allow


class ModelPreferences(BaseModel):
    """
    The server's preferences for model selection, requested by the client during
    sampling.

    Because LLMs can vary along multiple dimensions, choosing the "best" model is
    rarely straightforward.  Different models excel in different areas—some are
    faster but less capable, others are more capable but more expensive, and so
    on. This interface allows servers to express their priorities across multiple
    dimensions to help clients make an appropriate selection for their use case.

    These preferences are always advisory. The client MAY ignore them. It is also
    up to the client to decide how to interpret these preferences and how to
    balance them against other considerations.
    """

    hints: Optional[list[ModelHint]] = None
    """
    Optional hints to use for model selection.

    If multiple hints are specified, the client MUST evaluate them in order
    (such that the first match is taken).

    The client SHOULD prioritize these hints over the numeric priorities, but
    MAY still use the priorities to select from ambiguous matches.
    """

    costPriority: Optional[float] = None
    """
    How much to prioritize cost when selecting a model. A value of 0 means cost
    is not important, while a value of 1 means cost is the most important
    factor.
    """

    speedPriority: Optional[float] = None
    """
    How much to prioritize sampling speed (latency) when selecting a model. A
    value of 0 means speed is not important, while a value of 1 means speed is
    the most important factor.
    """

    intelligencePriority: Optional[float] = None
    """
    How much to prioritize intelligence and capabilities when selecting a
    model. A value of 0 means intelligence is not important, while a value of 1
    means intelligence is the most important factor.
    """

    class Config:
        extra = Extra.allow


class CreateMessageRequestParams(RequestParams):
    """Parameters for creating a message."""

    messages: list[SamplingMessage]
    modelPreferences: Optional[ModelPreferences] = None
    """
    The server's preferences for which model to select. The client MAY ignore
    these preferences.
    """
    systemPrompt: Optional[str] = None
    """An optional system prompt the server wants to use for sampling."""
    includeContext: Optional[IncludeContext] = None
    """
    A request to include context from one or more MCP servers (including the caller), to
    be attached to the prompt.
    """
    temperature: Optional[float] = None
    maxTokens: int
    """The maximum number of tokens to sample, as requested by the server."""
    stopSequences: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    """Optional metadata to pass through to the LLM provider."""

    class Config:
        extra = Extra.allow


class CreateMessageRequest(
    Request[CreateMessageRequestParams, Literal["sampling/createMessage"]]
):
    """A request from the server to sample an LLM via the client."""

    method: Literal["sampling/createMessage"] = "sampling/createMessage"
    params: CreateMessageRequestParams


StopReason = Union[Literal["endTurn", "stopSequence", "maxTokens"], str]


class CreateMessageResult(Result):
    """The client's response to a sampling/create_message request from the server."""

    role: Role
    content: Union[TextContent, ImageContent]
    model: str
    """The name of the model that generated the message."""
    stopReason: Optional[StopReason] = None
    """The reason why sampling stopped, if known."""


class ResourceReference(BaseModel):
    """A reference to a resource or resource template definition."""

    type: Literal["ref/resource"]
    uri: str
    """The URI or URI template of the resource."""

    class Config:
        extra = Extra.allow


class PromptReference(BaseModel):
    """Identifies a prompt."""

    type: Literal["ref/prompt"]
    name: str
    """The name of the prompt or prompt template"""

    class Config:
        extra = Extra.allow


class CompletionArgument(BaseModel):
    """The argument's information for completion requests."""

    name: str
    """The name of the argument"""
    value: str
    """The value of the argument to use for completion matching."""

    class Config:
        extra = Extra.allow


class CompleteRequestParams(RequestParams):
    """Parameters for completion requests."""

    ref: Union[ResourceReference, PromptReference]
    argument: CompletionArgument

    class Config:
        extra = Extra.allow


class CompleteRequest(Request[CompleteRequestParams, Literal["completion/complete"]]):
    """A request from the client to the server, to ask for completion options."""

    method: Literal["completion/complete"] = "completion/complete"
    params: CompleteRequestParams


class Completion(BaseModel):
    """Completion information."""

    values: list[str]
    """An array of completion values. Must not exceed 100 items."""
    total: Optional[int] = None
    """
    The total number of completion options available. This can exceed the number of
    values actually sent in the response.
    """
    hasMore: Optional[bool] = None
    """
    Indicates whether there are additional completion options beyond those provided in
    the current response, even if the exact total is unknown.
    """

    class Config:
        extra = Extra.allow


class CompleteResult(Result):
    """The server's response to a completion/complete request"""

    completion: Completion


class ListRootsRequest(Request[Optional[RequestParams], Literal["roots/list"]]):
    """
    Sent from the server to request a list of root URIs from the client. Roots allow
    servers to ask for specific directories or files to operate on. A common example
    for roots is providing a set of repositories or directories a server should operate
    on.

    This request is typically used when the server needs to understand the file system
    structure or access specific locations that the client has permission to read from.
    """

    method: Literal["roots/list"] = "roots/list"
    params: Optional[RequestParams] = None


class Root(BaseModel):
    """Represents a root directory or file that the server can operate on."""

    uri: FileUrl
    """
    The URI identifying the root. This *must* start with file:// for now.
    This restriction may be relaxed in future versions of the protocol to allow
    other URI schemes.
    """
    name: Optional[str] = None
    """
    An optional name for the root. This can be used to provide a human-readable
    identifier for the root, which may be useful for display purposes or for
    referencing the root in other parts of the application.
    """

    class Config:
        extra = Extra.allow


class ListRootsResult(Result):
    """
    The client's response to a roots/list request from the server.
    This result contains an array of Root objects, each representing a root directory
    or file that the server can operate on.
    """

    roots: list[Root]


class RootsListChangedNotification(
    Notification[Optional[NotificationParams], Literal["notifications/roots/list_changed"]]
):
    """
    A notification from the client to the server, informing it that the list of
    roots has changed.

    This notification should be sent whenever the client adds, removes, or
    modifies any root. The server should then request an updated list of roots
    using the ListRootsRequest.
    """

    method: Literal["notifications/roots/list_changed"] = "notifications/roots/list_changed"
    params: Optional[NotificationParams] = None


class CancelledNotificationParams(NotificationParams):
    """Parameters for cancellation notifications."""

    requestId: RequestId
    """The ID of the request to cancel."""
    reason: Optional[str] = None
    """An optional string describing the reason for the cancellation."""

    class Config:
        extra = Extra.allow


class CancelledNotification(
    Notification[CancelledNotificationParams, Literal["notifications/cancelled"]]
):
    """
    This notification can be sent by either side to indicate that it is canceling a
    previously-issued request.
    """

    method: Literal["notifications/cancelled"] = "notifications/cancelled"
    params: CancelledNotificationParams


class ClientRequest(BaseModel):
    __root__: Union[
        PingRequest,
        InitializeRequest,
        CompleteRequest,
        SetLevelRequest,
        GetPromptRequest,
        ListPromptsRequest,
        ListResourcesRequest,
        ListResourceTemplatesRequest,
        ReadResourceRequest,
        SubscribeRequest,
        UnsubscribeRequest,
        CallToolRequest,
        ListToolsRequest,
    ]


class ClientNotification(BaseModel):
    __root__: Union[
        CancelledNotification,
        ProgressNotification,
        InitializedNotification,
        RootsListChangedNotification,
    ]


class ClientResult(BaseModel):
    __root__: Union[EmptyResult, CreateMessageResult, ListRootsResult]


class ServerRequest(BaseModel):
    __root__: Union[PingRequest, CreateMessageRequest, ListRootsRequest]


class ServerNotification(BaseModel):
    __root__: Union[
        CancelledNotification,
        ProgressNotification,
        LoggingMessageNotification,
        ResourceUpdatedNotification,
        ResourceListChangedNotification,
        ToolListChangedNotification,
        PromptListChangedNotification,
    ]


class ServerResult(BaseModel):
    __root__: Union[
        EmptyResult,
        InitializeResult,
        CompleteResult,
        GetPromptResult,
        ListPromptsResult,
        ListResourcesResult,
        ListResourceTemplatesResult,
        ReadResourceResult,
        CallToolResult,
        ListToolsResult,
    ]
