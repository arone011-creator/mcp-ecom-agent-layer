"""The agent's HTTP surface. Sibling to server.py: two processes, one library.

server.py serves the MCP tools; this serves the agent that calls them.
They are deliberately separate processes rather than one, for two reasons
that are not about memory (the MCP service runs at 0.1GB of an 8GB limit):

  - only this one needs OPENAI_API_KEY, and co-locating would put the
    model key inside the process every customer's bearer token reaches;

  - approvals.py validates in the MCP server and the agent must never
    mint. Across two processes that is structural. In one it would be an
    import away from being false, guarded only by a test.

One turn is one SSE stream, and the stream carries TWO kinds of frame:

    event: assistant   a v1 event from contracts/assistant-events.v1.json,
                       forwarded to the browser by the bridge route
    event: control     {turn_id, session_id}, for the bridge route ONLY

The control frame is how the agent's MCP session id reaches the
storefront, which needs it to mint an approval against the right session
(M4 Task 5), without putting it in a stream that reaches a browser (M4
Task 4). SSE separates them at the transport, so the frozen event
contract does not have to change.
"""

import asyncio
import json
import subprocess
import traceback
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import config
from agent.events import OUT_OF_BAND
from agent.events import approval_required as approval_required_event
from agent.events import error as error_event
from agent.events import message_delta
from agent.history import UnsafeHistory, exportable_context, sanitise_history
from agent.loop import openai_model_call, run_turn, session_scoped_executor
from agent.titles import clean_title, name_conversation
from agent.tools import AGENT_TOOLS, list_openai_tools


class TurnRegistry:
    """Paused turns, findable by the request that resumes them.

    In process memory, so this service runs on ONE replica -- the same
    limit approvals.py documents for its spent-nonce set and InMemorySaver
    for its checkpoints. A paused turn also holds an open MCP session, so
    the approval deadline now bounds a real resource rather than an idle
    task.
    """

    def __init__(self) -> None:
        self._turns: dict[str, dict[str, Any]] = {}

    def open(self, session_id: str | None) -> str:
        turn_id = uuid.uuid4().hex
        self._turns[turn_id] = {
            "session_id": session_id,
            "future": asyncio.get_event_loop().create_future(),
        }
        return turn_id

    def session_id(self, turn_id: str) -> str | None:
        return self._turns[turn_id]["session_id"]

    def decide(self, turn_id: str, decision: dict) -> None:
        turn = self._turns[turn_id]

        if turn["future"].done():
            # A double-click must not resume a turn twice. The storefront
            # guards this too; neither side may rely on the other.
            raise ValueError("This turn has already been decided")

        turn["future"].set_result(decision)

    async def wait_for_decision(self, turn_id: str, timeout: float) -> dict:
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._turns[turn_id]["future"]), timeout=timeout
            )
        except asyncio.TimeoutError:
            # The same shape agent/loop.py::_decide produces, so the
            # loop's declined branch needs no new case.
            return {"approved": False, "reason": "expired"}

    def close(self, turn_id: str) -> None:
        self._turns.pop(turn_id, None)


registry = TurnRegistry()


def _build_sha() -> str:
    """The commit this container was built from.

    Returned by /health because a 200 proves A container is up, not that
    it is THIS one. Twice on this project a deploy was verified against
    the container it was replacing.
    """
    for source in (config.RAILWAY_GIT_COMMIT_SHA,):
        if source:
            return source[:12]

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()[:12]
        )
    except Exception:
        return "unknown"


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "sha": _build_sha(), "model": config.OPENAI_MODEL})


def _frame(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _unauthorised(reason: str) -> JSONResponse:
    return JSONResponse({"error": reason}, status_code=401)


def _check_service_key(request: Request) -> JSONResponse | None:
    """The gate that stops anyone else spending this project's credits.

    Checked before a single token is bought. An open endpoint that calls
    a paid model is a bill anyone can run up.
    """
    if not config.AGENT_SERVICE_KEY:
        return _unauthorised("This service is not configured to accept requests")

    if request.headers.get("x-agent-key") != config.AGENT_SERVICE_KEY:
        return _unauthorised("A valid X-Agent-Key is required")

    return None


def _customer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header[7:].strip() if header.lower().startswith("bearer ") else ""


async def turn(request: Request):
    refusal = _check_service_key(request)
    if refusal is not None:
        return refusal

    token = _customer_token(request)
    if not token:
        return _unauthorised("A customer bearer token is required")

    body = await request.json()
    utterance = (body.get("utterance") or "").strip()
    if not utterance:
        return JSONResponse({"error": "An utterance is required"}, status_code=400)

    # Checked HERE, before the response opens. Once the stream has begun
    # the status is already 200 and a refusal can only arrive as an error
    # event -- which is a worse answer to a request that was refusable
    # before a single token was spent.
    try:
        history = sanitise_history(body.get("history"))
    except UnsafeHistory:
        # Deliberately says nothing about which message or which role. The
        # value came out of the storefront's database, and echoing it back
        # describes that database to whoever is probing it.
        return JSONResponse({"error": "Replayed history was refused"}, status_code=400)

    return StreamingResponse(
        _stream_turn(utterance, token, history=history),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


async def _stream_turn(utterance: str, token: str, history: list[dict] | None = None):
    """One turn, as it happens.

    Events are published to a queue as the graph appends them rather than
    read off the returned state, because a stream that only arrives once
    the turn is over is not a stream -- and the pause in the middle is
    exactly when the customer most needs to see something.

    Three separate things have to be true for that, and for a while only
    two of them were, which is why the first live turn still arrived in
    one lump: this queue, run_turn publishing after each graph step
    rather than at the end, and the model call streaming its prose. Miss
    any one and the customer waits for the whole answer.

    The stream ends with a SECOND control frame carrying this turn's own
    messages, for the storefront to store against the conversation. It
    rides `control` and not `assistant` for the same reason the session id
    does: `assistant` is forwarded to the browser by exclusion, and the
    model transcript is not the browser's -- either to read or to send
    back.
    """
    queue: asyncio.Queue = asyncio.Queue()
    tools = await list_openai_tools(token, only=AGENT_TOOLS)

    async with session_scoped_executor(token) as session:
        turn_id = registry.open(session.session_id)

        yield _frame(
            "control", {"turn_id": turn_id, "session_id": session.session_id}
        )

        async def approve(request_payload: dict) -> dict:
            # The event the customer sees, then the wait. The storefront
            # mints against session_id from the control frame and POSTs
            # the decision back to /turn/{turn_id}/decision.
            await queue.put(
                approval_required_event(
                    -1,
                    request_payload["call_id"],
                    request_payload["tool"],
                    request_payload["arguments"],
                )
            )
            return await registry.wait_for_decision(
                turn_id, timeout=config.APPROVAL_WAIT_SECONDS
            )

        def on_delta(fragment: str) -> None:
            # Out of band and unnumbered: a rendering hint, not a fact
            # about the turn. The `message` event that follows carries the
            # whole answer and stays authoritative, so a reader that
            # ignores these -- an older storefront, say -- behaves exactly
            # as it did before fragments existed.
            queue.put_nowait(message_delta(fragment))

        async def drive():
            try:
                return await run_turn(
                    utterance,
                    model_call=openai_model_call(on_delta=on_delta),
                    execute_tool=session.execute,
                    tools=tools,
                    history=history,
                    approve=approve,
                    session_id=session.session_id,
                    on_event=queue.put_nowait,
                )
            except Exception:
                # A TURN THAT DIES MUST SAY SO. The response has already
                # begun -- 200, control frame, possibly half an answer --
                # so an exception here just stops the stream, and a stream
                # that stops cleanly with nothing in it is indistinguishable
                # from an assistant that had nothing to say. That is how a
                # broken deploy reached a customer as a blank panel.
                #
                # The exception is logged, never sent: a stack trace means
                # nothing to a shopper and can carry things a browser has
                # no business seeing.
                traceback.print_exc()
                queue.put_nowait(
                    error_event(
                        OUT_OF_BAND,
                        "The assistant ran into a problem and could not finish "
                        "that. Nothing was changed. Please try again.",
                        retryable=True,
                    )
                )
                return None
            finally:
                await queue.put(None)

        task = asyncio.create_task(drive())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _frame("assistant", event)

            # Awaited for cancellation and cleanup, and now for the return
            # value too. drive() swallows the failure deliberately, having
            # already reported it as an event -- raising here would abort
            # the stream a moment after telling the customer what went
            # wrong.
            state = await task

            # A turn that died answers None, and hands back nothing: its
            # messages may hold a tool_call nothing answered, which is a
            # shape the API refuses on the way back in. Storing that would
            # break every LATER turn of this conversation, not just this
            # one. No frame means the bridge stores null, and replay
            # starts after it.
            if state is not None:
                yield _frame("control", {"context": exportable_context(state)})
        finally:
            # An aborted stream must not leave the turn registered, or the
            # registry only ever grows.
            registry.close(turn_id)
            if not task.done():
                task.cancel()


async def decision(request: Request) -> JSONResponse:
    refusal = _check_service_key(request)
    if refusal is not None:
        return refusal

    turn_id = request.path_params["turn_id"]
    body = await request.json()

    try:
        registry.decide(
            turn_id,
            {
                "approved": bool(body.get("approved")),
                "token": body.get("token"),
            },
        )
    except KeyError:
        return JSONResponse({"error": "No such turn"}, status_code=404)
    except ValueError as already:
        return JSONResponse({"error": str(already)}, status_code=409)

    return JSONResponse({"ok": True})


async def title(request: Request) -> JSONResponse:
    """Name a conversation. One cheap model call, no tools, no session.

    ALWAYS 200 WITH A TITLE OR A NULL, never a 500 on a naming failure.
    The storefront's fallback is the customer's own first message, which
    is a perfectly good name -- so every way this can fail is an answer of
    "no title", not an error the caller has to special-case.
    """
    refusal = _check_service_key(request)
    if refusal is not None:
        return refusal

    body = await request.json()
    utterance = (body.get("utterance") or "").strip()
    if not utterance:
        return JSONResponse({"error": "An utterance is required"}, status_code=400)

    try:
        raw = await name_conversation(utterance, (body.get("answer") or "").strip())
    except Exception:
        # Logged, never returned. A stack trace means nothing to the
        # storefront, which keeps the fallback either way.
        traceback.print_exc()
        return JSONResponse({"title": None})

    return JSONResponse({"title": clean_title(raw)})


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/turn", turn, methods=["POST"]),
        Route("/turn/{turn_id}/decision", decision, methods=["POST"]),
        Route("/title", title, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    # Fail at boot rather than at the first request, and the same call
    # server.py makes about MCP_APPROVAL_SECRET: better loudly broken
    # than quietly open to anyone who finds the URL.
    if not config.AGENT_SERVICE_KEY:
        raise SystemExit("AGENT_SERVICE_KEY is required")

    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
