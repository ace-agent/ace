"""MCP Server for ACE Platform.

This module provides the MCP server entry point that exposes playbook
management tools to LLM clients (like Claude). It uses FastMCP for
simplified tool registration and supports SSE/stdio transports.

Configuration is loaded from environment variables:
- MCP_SERVER_HOST: Server bind host (default: 0.0.0.0)
- MCP_SERVER_PORT: Server port (default: 8001)
- ACE_API_KEY: API key for authentication (can also be passed per-tool)

Authentication priority (first match wins):
1. Explicit `api_key` parameter passed to tool
2. `X-API-Key` header (for SSE/HTTP transport)
3. `Authorization: Bearer <token>` header (for SSE/HTTP transport)
4. `ACE_API_KEY` environment variable (for stdio transport)
"""

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from ace_platform.config import get_settings
from ace_platform.core.api_keys import authenticate_api_key_async
from ace_platform.core.rate_limit import RATE_LIMITS, RateLimiter
from ace_platform.core.validation import (
    MAX_PLAYBOOK_DESCRIPTION_SIZE,
    MAX_PLAYBOOK_NAME_SIZE,
    validate_outcome_inputs,
    validate_playbook_content,
    validate_size,
)
from ace_platform.db.models import (
    Outcome,
    OutcomeStatus,
    Playbook,
    PlaybookSource,
    PlaybookStatus,
    PlaybookVersion,
)
from ace_platform.db.session import AsyncSessionLocal, close_async_db

settings = get_settings()

# Context variable for storing API key extracted from HTTP headers
# This allows tools to access the API key without it being passed as a parameter
_request_api_key: ContextVar[str | None] = ContextVar("request_api_key", default=None)


class HeaderAuthMiddleware:
    """ASGI middleware to extract API key from HTTP headers.

    This middleware extracts the API key from incoming requests and stores
    it in a context variable that can be accessed by MCP tools. It supports
    two header formats:

    1. X-API-Key: <api_key>
    2. Authorization: Bearer <api_key>

    The extracted key is available via the _request_api_key context variable
    and is automatically checked by get_api_key() in tool handlers.

    Example Claude Code MCP configuration with headers:
        {
            "mcpServers": {
                "ace": {
                    "type": "sse",
                    "url": "https://aceagent.io/mcp/sse",
                    "headers": {
                        "X-API-Key": "your-api-key-here"
                    }
                }
            }
        }
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application to wrap.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process the ASGI request.

        For HTTP requests, extracts the API key from headers and stores it
        in a context variable before calling the wrapped application.

        Args:
            scope: The ASGI scope dictionary.
            receive: The receive callable.
            send: The send callable.
        """
        if scope["type"] == "http":
            # Extract headers (they come as a list of byte tuples)
            headers = dict(scope.get("headers", []))

            # Try X-API-Key header first
            api_key = headers.get(b"x-api-key", b"").decode("utf-8") or None

            # Fall back to Authorization: Bearer header
            if not api_key:
                auth_header = headers.get(b"authorization", b"").decode("utf-8")
                if auth_header.lower().startswith("bearer "):
                    api_key = auth_header[7:].strip()

            # Store in context variable for the duration of this request
            token = _request_api_key.set(api_key)
            try:
                await self.app(scope, receive, send)
            finally:
                _request_api_key.reset(token)
        else:
            # For non-HTTP scopes (websocket, lifespan), pass through
            await self.app(scope, receive, send)


@dataclass
class MCPContext:
    """Application context available during MCP requests."""

    db: AsyncSession


@asynccontextmanager
async def mcp_lifespan(server: FastMCP) -> AsyncIterator[MCPContext]:
    """Manage MCP server lifecycle.

    Initializes database connection on startup and cleans up on shutdown.
    """
    # Startup: create a session for the lifespan
    async with AsyncSessionLocal() as db:
        try:
            yield MCPContext(db=db)
        finally:
            pass

    # Shutdown: close database connections
    await close_async_db()


# Create the MCP server instance
# Configure transport security to allow connections from Docker/Kubernetes
_transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,  # Disable for containerized deployment
    allowed_hosts=["*"],  # Allow health checks from any host
    allowed_origins=["*"],  # Allow any origin
)

mcp = FastMCP(
    name="ACE Platform",
    lifespan=mcp_lifespan,
    host=settings.mcp_server_host,
    port=settings.mcp_server_port,
    transport_security=_transport_security,
)


# Health check endpoints for Docker/Kubernetes
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    """Health check endpoint - returns healthy if server is running."""
    return JSONResponse({"status": "healthy", "service": "ace-mcp"})


@mcp.custom_route("/ready", methods=["GET"])
async def ready_check(request: Request) -> Response:
    """Readiness check endpoint - verifies database connectivity."""
    try:
        from sqlalchemy import text

        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        return JSONResponse({"status": "ready", "database": "connected"})
    except Exception:
        return JSONResponse(
            {"status": "not_ready", "database": "disconnected"},
            status_code=503,
        )


# Helper to get database session from context
def get_db(ctx: Context) -> AsyncSession:
    """Get database session from MCP context."""
    return ctx.request_context.lifespan_context.db


def get_api_key(api_key_param: str | None = None) -> str | None:
    """Get API key from parameter, HTTP header, or environment variable.

    Priority (first match wins):
    1. Explicit parameter (if provided and non-empty)
    2. X-API-Key or Authorization header (via context variable)
    3. ACE_API_KEY environment variable

    Args:
        api_key_param: Optional API key passed as tool parameter.

    Returns:
        API key string or None if not found.
    """
    # 1. Check explicit parameter
    if api_key_param:
        return api_key_param

    # 2. Check header (stored in context variable by HeaderAuthMiddleware)
    header_api_key = _request_api_key.get()
    if header_api_key:
        return header_api_key

    # 3. Fall back to environment variable (for stdio transport)
    return os.environ.get("ACE_API_KEY")


@mcp.tool()
async def get_playbook(
    playbook_id: Annotated[str, "UUID of the playbook to retrieve"],
    ctx: Context,
    api_key: Annotated[
        str | None, "API key for authentication (optional if ACE_API_KEY env var is set)"
    ] = None,
    version: Annotated[int | None, "Specific version number to retrieve (default: current)"] = None,
    section: Annotated[str | None, "Filter to a specific section by heading"] = None,
) -> str:
    """Get a playbook's content by ID.

    Returns the playbook name, description, and version content.
    Optionally retrieve a specific version or filter to a section.
    Requires a valid API key with 'playbooks:read' scope.

    Args:
        playbook_id: UUID of the playbook to retrieve.
        api_key: API key for authentication (optional if ACE_API_KEY env var is set).
        version: Specific version number (default: current version).
        section: Filter content to section matching this heading.

    Returns:
        Playbook content as structured markdown text.
    """
    from ace_platform.db.models import PlaybookVersion

    db = get_db(ctx)

    # Get API key from parameter or environment
    resolved_api_key = get_api_key(api_key)
    if not resolved_api_key:
        return "Error: No API key provided. Set ACE_API_KEY environment variable or pass api_key parameter."

    # Authenticate
    auth_result = await authenticate_api_key_async(db, resolved_api_key)
    if not auth_result:
        return "Error: Invalid or revoked API key"

    api_key_record, user = auth_result

    # Check scope
    from ace_platform.core.api_keys import check_scope

    if not check_scope(api_key_record, "playbooks:read"):
        return "Error: API key lacks 'playbooks:read' scope"

    try:
        pb_uuid = UUID(playbook_id)
    except ValueError:
        return f"Error: Invalid playbook ID format: {playbook_id}"

    # Get playbook
    playbook = await db.get(Playbook, pb_uuid)
    if not playbook:
        return f"Error: Playbook {playbook_id} not found"

    # Verify ownership
    if playbook.user_id != user.id:
        return "Error: Access denied - playbook belongs to another user"

    # Get requested version
    content = ""
    version_info = ""

    if version is not None:
        # Get specific version by version_number
        result = await db.execute(
            select(PlaybookVersion).where(
                PlaybookVersion.playbook_id == pb_uuid,
                PlaybookVersion.version_number == version,
            )
        )
        playbook_version = result.scalar_one_or_none()
        if not playbook_version:
            return f"Error: Version {version} not found for playbook {playbook_id}"
        content = playbook_version.content
        version_info = f" (v{version})"
    else:
        # Get current version
        if playbook.current_version_id:
            await db.refresh(playbook, ["current_version"])
            if playbook.current_version:
                content = playbook.current_version.content
                version_info = f" (v{playbook.current_version.version_number})"

    # Filter by section if requested
    if section and content:
        content = _extract_section(content, section)
        if not content:
            return f"Error: Section '{section}' not found in playbook"

    return f"""# {playbook.name}{version_info}

{playbook.description or "No description"}

---

{content or "No content yet - add outcomes to evolve the playbook."}
"""


def _extract_section(content: str, section_name: str) -> str:
    """Extract a specific section from markdown content.

    Finds a section by its heading (any level) and returns all content
    until the next heading of the same or higher level.

    Args:
        content: Full markdown content.
        section_name: Section heading to find (case-insensitive).

    Returns:
        Section content including the heading, or empty string if not found.
    """
    lines = content.split("\n")
    result_lines = []
    in_section = False
    section_level = 0

    section_name_lower = section_name.lower().strip()

    for line in lines:
        # Check if this is a heading
        if line.startswith("#"):
            # Count heading level
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip().lower()

            if in_section:
                # Check if we've hit another heading at same or higher level
                if level <= section_level:
                    break  # End of section
            elif heading_text == section_name_lower or section_name_lower in heading_text:
                # Found our section
                in_section = True
                section_level = level

        if in_section:
            result_lines.append(line)

    return "\n".join(result_lines).strip()


@mcp.tool()
async def list_playbooks(
    ctx: Context,
    api_key: Annotated[
        str | None, "API key for authentication (optional if ACE_API_KEY env var is set)"
    ] = None,
) -> str:
    """List all playbooks for the authenticated user.

    Returns a list of playbook names and IDs.
    Requires a valid API key with 'playbooks:read' scope.
    """
    db = get_db(ctx)

    # Get API key from parameter or environment
    resolved_api_key = get_api_key(api_key)
    if not resolved_api_key:
        return "Error: No API key provided. Set ACE_API_KEY environment variable or pass api_key parameter."

    # Authenticate
    auth_result = await authenticate_api_key_async(db, resolved_api_key)
    if not auth_result:
        return "Error: Invalid or revoked API key"

    api_key_record, user = auth_result

    # Check scope
    from ace_platform.core.api_keys import check_scope

    if not check_scope(api_key_record, "playbooks:read"):
        return "Error: API key lacks 'playbooks:read' scope"

    # Query user's playbooks
    result = await db.execute(
        select(Playbook).where(Playbook.user_id == user.id).order_by(Playbook.created_at.desc())
    )
    playbooks = result.scalars().all()

    if not playbooks:
        return "No playbooks found. Create one in the dashboard first."

    lines = ["# Your Playbooks\n"]
    for pb in playbooks:
        lines.append(f"- **{pb.name}** (`{pb.id}`)")
        if pb.description:
            lines.append(f"  {pb.description[:100]}...")

    return "\n".join(lines)


@mcp.tool()
async def create_playbook(
    name: Annotated[str, "Name for the playbook (max 255 chars)"],
    ctx: Context,
    api_key: Annotated[
        str | None, "API key for authentication (optional if ACE_API_KEY env var is set)"
    ] = None,
    description: Annotated[str | None, "Description of the playbook (max 2000 chars)"] = None,
    initial_content: Annotated[
        str | None, "Initial playbook content in markdown (max 100KB)"
    ] = None,
) -> str:
    """Create a new playbook.

    Creates a new playbook with optional initial content. If initial_content
    is provided, version 1 is created automatically.

    Requires a valid API key with 'playbooks:write' scope.

    Size limits:
    - name: 255 characters max
    - description: 2KB max
    - initial_content: 100KB max

    Args:
        name: Name for the playbook.
        api_key: API key for authentication (optional if ACE_API_KEY env var is set).
        description: Optional description of the playbook.
        initial_content: Optional initial content in markdown format.

    Returns:
        Success message with playbook ID, or error message.
    """
    from ace_platform.core.api_keys import check_scope
    from ace_platform.core.limits import SubscriptionTier, get_tier_limits

    db = get_db(ctx)

    # Get API key from parameter or environment
    resolved_api_key = get_api_key(api_key)
    if not resolved_api_key:
        return "Error: No API key provided. Set ACE_API_KEY environment variable or pass api_key parameter."

    # Authenticate
    auth_result = await authenticate_api_key_async(db, resolved_api_key)
    if not auth_result:
        return "Error: Invalid or revoked API key"

    api_key_record, user = auth_result

    # Check scope
    if not check_scope(api_key_record, "playbooks:write"):
        return "Error: API key lacks 'playbooks:write' scope"

    # Validate inputs
    error = validate_size(name, "Name", MAX_PLAYBOOK_NAME_SIZE)
    if error:
        return f"Error: {error}"

    if not name or not name.strip():
        return "Error: Name is required and cannot be empty"

    if description:
        error = validate_size(description, "Description", MAX_PLAYBOOK_DESCRIPTION_SIZE)
        if error:
            return f"Error: {error}"

    if initial_content:
        error = validate_playbook_content(initial_content)
        if error:
            return f"Error: {error}"

    # Check max_playbooks limit for user's tier
    user_tier = (
        SubscriptionTier(user.subscription_tier)
        if user.subscription_tier
        else SubscriptionTier.FREE
    )
    limits = get_tier_limits(user_tier)

    if limits.max_playbooks is not None:
        # Count existing playbooks
        count_query = select(func.count()).select_from(
            select(Playbook).where(Playbook.user_id == user.id).subquery()
        )
        current_count = await db.scalar(count_query) or 0

        if current_count >= limits.max_playbooks:
            return (
                f"Error: You have reached the maximum number of playbooks ({limits.max_playbooks}) "
                f"for your {user_tier.value} subscription. Please upgrade to create more playbooks."
            )

    # Create playbook
    playbook = Playbook(
        user_id=user.id,
        name=name.strip(),
        description=description.strip() if description else None,
        status=PlaybookStatus.ACTIVE,
        source=PlaybookSource.USER_CREATED,
    )
    db.add(playbook)
    await db.flush()

    # Create initial version if content provided
    version_info = ""
    if initial_content:
        # Count ACE-format bullets: [id] helpful=X harmful=Y :: content
        ace_bullet_pattern = r"\[[^\]]+\]\s*helpful=\d+\s*harmful=\d+\s*::"
        bullet_count = len(re.findall(ace_bullet_pattern, initial_content))

        version = PlaybookVersion(
            playbook_id=playbook.id,
            version_number=1,
            content=initial_content,
            bullet_count=bullet_count,
        )
        db.add(version)
        await db.flush()

        playbook.current_version_id = version.id
        version_info = f" with version 1 ({bullet_count} bullets)"

    await db.commit()

    return f"Playbook created successfully{version_info} (ID: {playbook.id})"


@mcp.tool()
async def create_version(
    playbook_id: Annotated[str, "UUID of the playbook to create a version for"],
    content: Annotated[str, "New version content in markdown (max 100KB)"],
    ctx: Context,
    api_key: Annotated[
        str | None, "API key for authentication (optional if ACE_API_KEY env var is set)"
    ] = None,
    diff_summary: Annotated[str | None, "Brief description of changes (max 500 chars)"] = None,
) -> str:
    """Create a new version of a playbook.

    Creates an immutable version with incremented version number.
    The playbook's current_version is updated to point to the new version.

    Requires a valid API key with 'playbooks:write' scope.

    Size limits:
    - content: 100KB max
    - diff_summary: 500 characters max

    Args:
        playbook_id: UUID of the playbook to update.
        content: New version content in markdown format.
        api_key: API key for authentication (optional if ACE_API_KEY env var is set).
        diff_summary: Optional brief description of what changed.

    Returns:
        Success message with version number, or error message.
    """
    from ace_platform.core.api_keys import check_scope

    db = get_db(ctx)

    # Get API key from parameter or environment
    resolved_api_key = get_api_key(api_key)
    if not resolved_api_key:
        return "Error: No API key provided. Set ACE_API_KEY environment variable or pass api_key parameter."

    # Authenticate
    auth_result = await authenticate_api_key_async(db, resolved_api_key)
    if not auth_result:
        return "Error: Invalid or revoked API key"

    api_key_record, user = auth_result

    # Check scope
    if not check_scope(api_key_record, "playbooks:write"):
        return "Error: API key lacks 'playbooks:write' scope"

    # Validate playbook ID
    try:
        pb_uuid = UUID(playbook_id)
    except ValueError:
        return f"Error: Invalid playbook ID format: {playbook_id}"

    # Validate inputs
    if not content or not content.strip():
        return "Error: Content is required and cannot be empty"

    error = validate_playbook_content(content)
    if error:
        return f"Error: {error}"

    if diff_summary:
        error = validate_size(diff_summary, "Diff summary", 500)
        if error:
            return f"Error: {error}"

    # Get playbook and verify ownership
    playbook = await db.get(Playbook, pb_uuid)
    if not playbook:
        return f"Error: Playbook {playbook_id} not found"

    if playbook.user_id != user.id:
        return "Error: Access denied - playbook belongs to another user"

    # Calculate bullet count (done once, outside retry loop)
    ace_bullet_pattern = r"\[[^\]]+\]\s*helpful=\d+\s*harmful=\d+\s*::"
    bullet_count = len(re.findall(ace_bullet_pattern, content))

    # Retry loop to handle race conditions on version_number
    max_retries = 3
    for attempt in range(max_retries):
        # Get current max version number
        max_version_query = select(func.max(PlaybookVersion.version_number)).where(
            PlaybookVersion.playbook_id == pb_uuid
        )
        current_max = await db.scalar(max_version_query) or 0
        new_version_number = current_max + 1

        # Create new version
        version = PlaybookVersion(
            playbook_id=pb_uuid,
            version_number=new_version_number,
            content=content,
            bullet_count=bullet_count,
            diff_summary=diff_summary.strip() if diff_summary else None,
            created_by_job_id=None,  # Manual edit, not from evolution job
        )
        db.add(version)

        try:
            await db.flush()
            # Update playbook to point to new version
            playbook.current_version_id = version.id
            await db.commit()

            return (
                f"Version {new_version_number} created successfully "
                f"({bullet_count} bullets) (ID: {version.id})"
            )
        except IntegrityError:
            # Race condition: another request created this version number
            # Rollback and retry with a fresh version number
            await db.rollback()
            # Re-fetch playbook after rollback (session state is cleared)
            playbook = await db.get(Playbook, pb_uuid)
            if not playbook:
                return f"Error: Playbook {playbook_id} not found after rollback"
            if attempt == max_retries - 1:
                return "Error: Failed to create version due to concurrent modification. Please try again."

    # This should never be reached due to the return in the loop
    return "Error: Unexpected error creating version"


@mcp.tool()
async def record_outcome(
    playbook_id: Annotated[str, "UUID of the playbook this outcome is for"],
    task_description: Annotated[str, "Description of the task that was attempted"],
    outcome: Annotated[str, "Outcome status: 'success', 'failure', or 'partial'"],
    ctx: Context,
    api_key: Annotated[
        str | None, "API key for authentication (optional if ACE_API_KEY env var is set)"
    ] = None,
    notes: Annotated[str | None, "Optional notes about the outcome"] = None,
    reasoning_trace: Annotated[str | None, "Optional reasoning trace/log"] = None,
) -> str:
    """Record a task outcome for playbook evolution.

    After recording enough outcomes, the playbook will automatically evolve
    to incorporate lessons learned. Requires 'outcomes:write' scope and
    email verification.

    Size limits:
    - task_description: 10KB max
    - notes: 2KB max
    - reasoning_trace: 10KB max
    """
    # Validate input sizes
    validation_error = validate_outcome_inputs(task_description, notes, reasoning_trace)
    if validation_error:
        return f"Error: {validation_error}"

    db = get_db(ctx)

    # Get API key from parameter or environment
    resolved_api_key = get_api_key(api_key)
    if not resolved_api_key:
        return "Error: No API key provided. Set ACE_API_KEY environment variable or pass api_key parameter."

    # Authenticate
    auth_result = await authenticate_api_key_async(db, resolved_api_key)
    if not auth_result:
        return "Error: Invalid or revoked API key"

    api_key_record, user = auth_result

    # Check email verification (required to prevent abuse)
    if not user.email_verified:
        return "Error: Email verification required. Please verify your email before recording outcomes."

    # Check scope
    from ace_platform.core.api_keys import check_scope

    if not check_scope(api_key_record, "outcomes:write"):
        return "Error: API key lacks 'outcomes:write' scope"

    try:
        pb_uuid = UUID(playbook_id)
    except ValueError:
        return f"Error: Invalid playbook ID format: {playbook_id}"

    # Validate outcome status
    try:
        outcome_status = OutcomeStatus(outcome.lower())
    except ValueError:
        return f"Error: Invalid outcome status '{outcome}'. Use 'success', 'failure', or 'partial'."

    # Get playbook and verify ownership
    playbook = await db.get(Playbook, pb_uuid)
    if not playbook:
        return f"Error: Playbook {playbook_id} not found"

    if playbook.user_id != user.id:
        return "Error: Access denied - playbook belongs to another user"

    # Create outcome record
    new_outcome = Outcome(
        playbook_id=pb_uuid,
        task_description=task_description,
        outcome_status=outcome_status,
        notes=notes,
        reasoning_trace=reasoning_trace,
    )
    db.add(new_outcome)
    await db.commit()

    return f"Outcome recorded successfully (ID: {new_outcome.id}). Status: {outcome_status.value}"


@mcp.tool()
async def get_evolution_status(
    job_id: Annotated[str, "UUID of the evolution job to check"],
    ctx: Context,
    api_key: Annotated[
        str | None, "API key for authentication (optional if ACE_API_KEY env var is set)"
    ] = None,
) -> str:
    """Get the status of an evolution job.

    Returns the job status, progress, timing, and any error information.
    Requires a valid API key with 'evolution:read' scope.

    Use this to poll for job completion after triggering evolution.

    Args:
        job_id: UUID of the evolution job to check.
        api_key: API key for authentication (optional if ACE_API_KEY env var is set).

    Returns:
        Job status information as structured text.
    """
    from ace_platform.db.models import EvolutionJob

    db = get_db(ctx)

    # Get API key from parameter or environment
    resolved_api_key = get_api_key(api_key)
    if not resolved_api_key:
        return "Error: No API key provided. Set ACE_API_KEY environment variable or pass api_key parameter."

    # Authenticate
    auth_result = await authenticate_api_key_async(db, resolved_api_key)
    if not auth_result:
        return "Error: Invalid or revoked API key"

    api_key_record, user = auth_result

    # Check scope
    from ace_platform.core.api_keys import check_scope

    if not check_scope(api_key_record, "evolution:read"):
        return "Error: API key lacks 'evolution:read' scope"

    try:
        job_uuid = UUID(job_id)
    except ValueError:
        return f"Error: Invalid job ID format: {job_id}"

    # Get evolution job
    job = await db.get(EvolutionJob, job_uuid)
    if not job:
        return f"Error: Evolution job {job_id} not found"

    # Get the associated playbook to verify ownership
    playbook = await db.get(Playbook, job.playbook_id)
    if not playbook or playbook.user_id != user.id:
        return "Error: Access denied - job belongs to another user"

    # Format timestamps
    started = job.started_at.isoformat() if job.started_at else "Not started"
    completed = job.completed_at.isoformat() if job.completed_at else "Not completed"

    # Build response
    lines = [
        "# Evolution Job Status",
        "",
        f"**Job ID:** {job.id}",
        f"**Status:** {job.status.value}",
        f"**Playbook:** {playbook.name} (`{job.playbook_id}`)",
        f"**Outcomes Processed:** {job.outcomes_processed}",
        "",
        "## Timing",
        f"- **Created:** {job.created_at.isoformat()}",
        f"- **Started:** {started}",
        f"- **Completed:** {completed}",
    ]

    # Add error message if present
    if job.error_message:
        lines.extend(
            [
                "",
                "## Error",
                "```",
                f"{job.error_message}",
                "```",
            ]
        )

    # Add version info if available
    if job.from_version_id or job.to_version_id:
        lines.extend(["", "## Versions"])
        if job.from_version_id:
            lines.append(f"- **From Version:** {job.from_version_id}")
        if job.to_version_id:
            lines.append(f"- **To Version:** {job.to_version_id}")

    return "\n".join(lines)


@mcp.tool()
async def trigger_evolution(
    playbook_id: Annotated[str, "UUID of the playbook to evolve"],
    ctx: Context,
    api_key: Annotated[
        str | None, "API key for authentication (optional if ACE_API_KEY env var is set)"
    ] = None,
) -> str:
    """Manually trigger playbook evolution.

    This queues an evolution job that will process unprocessed outcomes
    and generate an improved playbook version. Requires 'evolution:write' scope
    and email verification.

    Note: Evolution happens automatically based on thresholds, but you can
    trigger it manually if needed.
    """
    db = get_db(ctx)

    # Get API key from parameter or environment
    resolved_api_key = get_api_key(api_key)
    if not resolved_api_key:
        return "Error: No API key provided. Set ACE_API_KEY environment variable or pass api_key parameter."

    # Authenticate
    auth_result = await authenticate_api_key_async(db, resolved_api_key)
    if not auth_result:
        return "Error: Invalid or revoked API key"

    api_key_record, user = auth_result

    # Check email verification (required for evolution to prevent abuse)
    if not user.email_verified:
        return "Error: Email verification required. Please verify your email before triggering evolutions."

    # Check scope
    from ace_platform.core.api_keys import check_scope

    if not check_scope(api_key_record, "evolution:write"):
        return "Error: API key lacks 'evolution:write' scope"

    try:
        pb_uuid = UUID(playbook_id)
    except ValueError:
        return f"Error: Invalid playbook ID format: {playbook_id}"

    # Get playbook and verify ownership
    playbook = await db.get(Playbook, pb_uuid)
    if not playbook:
        return f"Error: Playbook {playbook_id} not found"

    if playbook.user_id != user.id:
        return "Error: Access denied - playbook belongs to another user"

    # Check rate limit (10/hour per playbook)
    limiter = RateLimiter()
    try:
        config = RATE_LIMITS["evolution"]
        rate_result = await limiter.is_allowed(
            "evolution",
            playbook_id,
            limit=config["limit"],
            window_seconds=config["window_seconds"],
        )
        if not rate_result.allowed:
            return f"Error: Rate limit exceeded. Evolution can be triggered at most {config['limit']} times per hour. Try again later."
    except Exception:
        # If Redis unavailable, allow the request
        pass

    # Check spending/evolution limits
    from ace_platform.core.limits import SubscriptionTier, check_can_evolve

    # Get user's subscription tier (defaults to FREE if not set)
    user_tier = (
        SubscriptionTier(user.subscription_tier)
        if user.subscription_tier
        else SubscriptionTier.FREE
    )
    can_proceed, error_message = await check_can_evolve(
        db, user.id, user_tier, has_payment_method=user.has_payment_method
    )
    if not can_proceed:
        # Provide a helpful message with link for payment method requirement
        if "payment method" in error_message.lower():
            from ace_platform.config import get_settings

            settings = get_settings()
            # Get frontend URL for the card setup link
            frontend_url = settings.frontend_url or "https://app.aceagent.io"
            setup_url = f"{frontend_url}/pricing"

            return (
                f"Evolution blocked: A payment method is required to trigger evolutions.\n\n"
                f"Add a card to unlock evolution triggers: {setup_url}\n\n"
                f"Once you've added a payment method, you can trigger evolutions on your playbooks."
            )
        return f"Error: {error_message}"

    # Trigger evolution
    from ace_platform.core.evolution_jobs import trigger_evolution_async

    try:
        result = await trigger_evolution_async(db, pb_uuid)
        await db.commit()

        if result.is_new:
            return f"Evolution job queued (Job ID: {result.job_id}). Check back later for results."
        else:
            return f"Evolution already in progress (Job ID: {result.job_id}, Status: {result.status.value})."
    except ValueError as e:
        return f"Error: {e}"


def run_server(transport: str = "stdio") -> None:
    """Run the MCP server.

    Args:
        transport: Transport to use ('stdio' or 'sse').
                   Use 'stdio' for local development with Claude Desktop.
                   Use 'sse' for web-based clients.
    """
    # Host and port are configured at FastMCP initialization
    mcp.run(transport=transport)


if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    run_server(transport)
