import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import handler, { upstreamUrl } from "../../gateway/proxy.js";
import { Agent, fetch, getGlobalDispatcher, setGlobalDispatcher } from "../../gateway/node_modules/undici/index.js";

let upstream, proxy, base, binding;
const previous = process.env.APP_URL;
const listen = async server => { server.listen(0, "127.0.0.1"); await once(server, "listening"); return `http://127.0.0.1:${server.address().port}`; };

before(async () => {
  upstream = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    if (req.url.startsWith("/api/slow")) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ complete: true }));
    } else if (req.url.startsWith("/api/failure")) {
      res.writeHead(422, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ detail: "Invalid configuration" }));
    } else if (req.url.startsWith("/api/redirect")) {
      res.writeHead(307, { Location: binding + "/api/health" }); res.end();
    } else if (req.url.startsWith("/api/stream")) {
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write("data: first\n\n"); setTimeout(() => res.end("data: second\n\n"), 20);
    } else {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ path: req.url, method: req.method, body: Buffer.concat(chunks).toString(), authorization: req.headers.authorization, host: req.headers.host }));
    }
  });
  binding = await listen(upstream);
  proxy = createServer((req, res) => { handler(req, res); });
  base = await listen(proxy);
});
after(async () => {
  if (previous === undefined) delete process.env.APP_URL; else process.env.APP_URL = previous;
  await Promise.all([new Promise(resolve => proxy.close(resolve)), new Promise(resolve => upstream.close(resolve))]);
});

test("missing runtime binding fails clearly, with no hardcoded fallback", async () => {
  delete process.env.APP_URL;
  const result = await fetch(base + "/api/health");
  assert.equal(result.status, 503);
  assert.match((await result.json()).detail, /APP_URL/);
});
test("binding is read at invocation time, and /api and query are preserved", async () => {
  process.env.APP_URL = binding;
  const result = await fetch(base + "/api/health?sample=3");
  assert.equal(result.status, 200);
  assert.equal((await result.json()).path, "/api/health?sample=3");
  assert.equal(result.headers.get("cache-control"), "no-store");
});
test("POST payload, method and authorization survive forwarding", async () => {
  process.env.APP_URL = binding;
  const body = JSON.stringify({ batch_size: 3, defense_type: "sparse" });
  const result = await fetch(base + "/api/run-defense", { method: "POST", body, headers: { "Content-Type": "application/json", authorization: "Bearer fixture" } });
  const data = await result.json();
  assert.equal(data.method, "POST"); assert.equal(data.body, body);
  assert.equal(data.authorization, "Bearer fixture");
  assert.equal(data.host, new URL(binding).host);
});
test("backend validation status and JSON are not replaced by an HTML page", async () => {
  const result = await fetch(base + "/api/failure", { method: "POST", body: "{}" });
  assert.equal(result.status, 422);
  assert.equal((await result.json()).detail, "Invalid configuration");
});
test("redirects remain on the public origin", async () => {
  const result = await fetch(base + "/api/redirect", { redirect: "manual" });
  assert.equal(result.status, 307); assert.equal(result.headers.get("location"), "/api/health");
});
test("streaming responses are forwarded", async () => {
  const result = await fetch(base + "/api/stream");
  assert.match(result.headers.get("content-type"), /event-stream/);
  assert.equal(await result.text(), "data: first\n\ndata: second\n\n");
});
test("request hosts cannot replace the binding's origin", () => {
  assert.equal(upstreamUrl("https://untrusted.invalid/api/health", binding).origin, binding);
  assert.throws(() => upstreamUrl("/elsewhere", binding), /API path/);
  assert.throws(() => upstreamUrl("/api/health", "file:///tmp/app"), /protocol/);
});

test("slow response headers use the experiment dispatcher rather than the global timeout", async () => {
  const original = getGlobalDispatcher();
  const shortWait = new Agent({ headersTimeout: 30 });
  const client = new Agent({ headersTimeout: 5000 });
  process.env.APP_URL = binding;
  setGlobalDispatcher(shortWait);
  try {
    await assert.rejects(fetch(binding + "/api/slow"), error => error.cause?.code === "UND_ERR_HEADERS_TIMEOUT");
    const response = await fetch(base + "/api/slow", { dispatcher: client });
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { complete: true });
  } finally {
    setGlobalDispatcher(original);
    await Promise.all([shortWait.close(), client.close()]);
  }
});
