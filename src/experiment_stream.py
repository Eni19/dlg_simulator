"""Stream telemetry and the final result over one HTTP invocation (no job store)."""
import asyncio
import json
import threading
import math
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from .progress import reporter, report

# Torch RNG is process-global: serialize scientific jobs rather than interleave seeds.
experiment_lock = threading.Lock()


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


async def stream_experiment(operation, allow_queue=True):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    disconnected = threading.Event()

    def send(event):
        if not disconnected.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker():
        token = reporter.set(send)
        acquired = False
        try:
            report("waiting", "Aguardando disponibilidade do backend")
            acquired = experiment_lock.acquire(blocking=allow_queue)
            if not acquired:
                raise HTTPException(status_code=429, detail="Esta instância da demonstração já está calculando um experimento. Tente novamente em alguns instantes.")
            if disconnected.is_set():
                return
            report("preparing", "Preparando o experimento")
            result = asyncio.run(operation())
            send({"type": "result", "result": jsonable_encoder(result)})
        except Exception as error:
            send({"type": "error", "detail": str(error.detail) if isinstance(error, HTTPException) else str(error),
                  "status": error.status_code if isinstance(error, HTTPException) else 500})
        finally:
            if acquired:
                experiment_lock.release()
            reporter.reset(token)

    task = asyncio.create_task(asyncio.to_thread(worker))
    # Retrieve exceptions even if the browser disconnects before the worker ends.
    task.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
    try:
        yield "\n"  # Flush headers immediately, before any data preparation.
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=10)
            except TimeoutError:
                yield "\n"  # Keep proxies alive without inventing progress.
                continue
            yield json.dumps(safe_json(event), ensure_ascii=False, allow_nan=False) + "\n"
            if event["type"] in {"result", "error"}:
                break
    finally:
        disconnected.set()
