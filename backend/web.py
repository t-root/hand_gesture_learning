"""FastAPI: route HTTP/WebSocket và mount thư mục exports."""

import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .app_state import AppState
from .paths import EXPORT_DIR, STATIC_INDEX

STATE = AppState()

app = FastAPI(title='Hand Gesture Learning Server')
app.mount('/exports', StaticFiles(directory=str(EXPORT_DIR)), name='exports')


@app.get('/', response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(STATIC_INDEX.read_text(encoding='utf-8'))


@app.get('/api/status')
async def api_status() -> JSONResponse:
    return JSONResponse(STATE.snapshot())


@app.websocket('/ws')
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()

    async def push_event(payload: dict) -> None:
        await websocket.send_json(payload)

    STATE.attach_websocket(loop, push_event)
    await websocket.send_json({'type': 'snapshot', 'snapshot': STATE.snapshot()})
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                await websocket.send_json({'type': 'error', 'error': 'invalid_json'})
                continue
            mtype = str(msg.get('type') or '')
            if mtype == 'frame':
                cfg = msg.get('config')
                cfg_dict = cfg if isinstance(cfg, dict) else None
                result = STATE.process_frame(str(msg.get('image') or ''), cfg_dict)
                await websocket.send_json({'type': 'frame_result', **result})
            else:
                result = STATE.handle_command(msg)
                await websocket.send_json({'type': 'command_result', **result, 'command': mtype})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_json({'type': 'error', 'error': str(exc)})
        except Exception:
            pass
    finally:
        STATE.stop_camera()
        STATE.detach_websocket()
