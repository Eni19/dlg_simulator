import asyncio
import json
import unittest
from src.progress import report, reporter
from src.experiment_stream import stream_experiment, experiment_lock


class ProgressStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_showcase_instance_does_not_queue_more_work(self):
        async def operation():
            self.fail("Busy instance must not execute another operation")
        experiment_lock.acquire()
        try:
            events = [json.loads(line) async for line in stream_experiment(operation, allow_queue=False) if line.strip()]
        finally:
            experiment_lock.release()
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["status"], 429)

    async def test_counts_arrive_before_result_and_context_is_isolated(self):
        async def operation():
            for cycle in range(1, 4):
                report("optimizing", "fixture", cycle, 3, 1 / cycle)
            return {"success": True, "metric": float("nan")}

        events = []
        async for line in stream_experiment(operation):
            if line.strip():
                events.append(json.loads(line))
        counts = [e["current"] for e in events if e.get("phase") == "optimizing"]
        self.assertEqual(counts, [1, 2, 3])
        self.assertEqual(events[-1]["type"], "result")
        self.assertIsNone(events[-1]["result"]["metric"])
        self.assertIsNone(reporter.get())

    async def test_failure_is_terminal_error_not_success(self):
        async def operation():
            report("preparing", "Loading")
            raise RuntimeError("fixture failed")
        events = [json.loads(line) async for line in stream_experiment(operation) if line.strip()]
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["detail"], "fixture failed")
        self.assertFalse(any(e["type"] == "result" for e in events))

    async def test_reports_do_not_wait_for_the_final_result(self):
        release = __import__("threading").Event()
        async def operation():
            report("optimizing", "fixture", 1, 10)
            await asyncio.to_thread(release.wait, 5)
            return {"success": True}
        stream = stream_experiment(operation)
        try:
            while True:
                line = await asyncio.wait_for(anext(stream), 2)
                if line.strip() and json.loads(line).get("phase") == "optimizing":
                    break
            self.assertFalse(release.is_set())
        finally:
            release.set()
            await stream.aclose()
