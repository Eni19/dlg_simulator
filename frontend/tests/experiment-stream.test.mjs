import { test } from "node:test";
import assert from "node:assert/strict";
import { runExperimentStream } from "../src/lib/api.ts";

const config = { environment: "centralized", defense: "none", batchSize: 1, numIters: 10, device: "cpu", defenseParams: {} };

async function withStream(text, callback) {
  const previous = globalThis.fetch;
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) {
      // Exercise partial JSON lines and UTF-8 characters split across network reads.
      for (const byte of new TextEncoder().encode(text)) controller.enqueue(new Uint8Array([byte]));
      controller.close();
    },
  }));
  try { await callback(); } finally { globalThis.fetch = previous; }
}

test("partial UTF-8 and JSON chunks yield real counts then mapped result", async () => {
  const event = { type: "progress", phase: "optimizing", text: "Otimização", current: 1, total: 10 };
  const result = { type: "result", result: { success: true, device: "cpu", loss_history: [1], batch_size: 1 } };
  await withStream("\n" + JSON.stringify(event) + "\n\n" + JSON.stringify(result), async () => {
    const events = [];
    const response = await runExperimentStream(config, value => events.push(value), new AbortController().signal);
    assert.deepEqual(events, [event]);
    assert.equal(response.success, true);
    assert.deepEqual(response.lossHistory, [1]);
  });
});

test("in-stream errors do not become successful results", async () => {
  await withStream('{"type":"error","detail":"Falha nos dados","status":503}\n', async () => {
    await assert.rejects(runExperimentStream(config, () => {}, new AbortController().signal), /Falha nos dados/);
  });
});

test("truncated stream cannot report completion", async () => {
  await withStream('{"type":"progress","current":1,"total":10}\n', async () => {
    await assert.rejects(runExperimentStream(config, () => {}, new AbortController().signal), /antes do resultado final/);
  });
});
