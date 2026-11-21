import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimitMiddleware:
    """
    Rejects request bodies larger than `max_bytes` with HTTP 413.

    Checks `Content-Length` up front and also counts streamed bytes, so chunked
    uploads without a length header are limited too. Works with any
    Starlette/FastAPI version.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _send_413(self, send: Send) -> None:
        body = json.dumps(
            {"detail": f"Request body exceeds the {self.max_bytes} byte limit."}
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                if declared is not None and declared > self.max_bytes:
                    await self._send_413(send)
                    return
                break

        received = 0
        rejected = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    if not response_started:
                        await self._send_413(send)
                    # Tell the app the client went away so it stops parsing.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return  # 413 already sent; drop the app's own error response.
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)
