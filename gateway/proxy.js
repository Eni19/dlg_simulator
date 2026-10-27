import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { Agent, fetch } from "undici";

// DLG does not send headers until optimization has finished. Scope this longer
// wait to our internal API calls, without changing Node's global dispatcher.
const experimentDispatcher = new Agent({
  headersTimeout: 30 * 60 * 1000,
  bodyTimeout: 30 * 60 * 1000,
});

const HOP_HEADERS = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
  "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
]);

function jsonError(res, status, detail) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify({ detail }));
}

/** Preserve the /api prefix: the FastAPI application owns /api/* routes. */
export function upstreamUrl(requestUrl, binding) {
  const base = new URL(binding);
  if (!["http:", "https:"].includes(base.protocol)) throw new Error("Invalid service binding protocol");
  const incoming = new URL(requestUrl, "http://request.invalid");
  if (!incoming.pathname.startsWith("/api/")) throw new Error("Invalid API path");
  // Use the trusted binding's origin, never an incoming hostname or forwarding header.
  base.pathname = incoming.pathname;
  base.search = incoming.search;
  base.hash = "";
  return base;
}

export default async function handler(req, res) {
  // Binding is resolved for each invocation. It is not embedded in the Vite build.
  const binding = process.env.APP_URL;
  if (!binding) {
    jsonError(res, 503, "The internal API binding APP_URL is unavailable.");
    return;
  }

  const controller = new AbortController();
  const abort = () => controller.abort();
  const disconnected = () => { if (!res.writableEnded) abort(); };
  req.once("aborted", abort);
  res.once("close", disconnected);

  try {
    const url = upstreamUrl(req.originalUrl || req.url || "/", binding);
    const headers = new Headers();
    const connectionTokens = String(req.headers.connection || "").toLowerCase().split(",").map(value => value.trim());
    for (const [key, value] of Object.entries(req.headers)) {
      if (value === undefined || HOP_HEADERS.has(key.toLowerCase()) || connectionTokens.includes(key.toLowerCase())) continue;
      headers.set(key, Array.isArray(value) ? value.join(", ") : value);
    }
    // Fetch decodes compressed responses; omit stale length/encoding headers below.
    headers.set("accept-encoding", "identity");
    const options = { method: req.method, headers, signal: controller.signal, redirect: "manual", dispatcher: experimentDispatcher };
    if (!["GET", "HEAD"].includes(req.method || "GET")) {
      // Vercel may supply a parsed body; plain Node/vercel dev may supply a stream.
      if (req.body !== undefined) {
        options.body = Buffer.isBuffer(req.body) || typeof req.body === "string" ? req.body : JSON.stringify(req.body);
      } else {
        options.body = req;
        options.duplex = "half";
      }
    }

    const upstream = await fetch(url, options);
    res.statusCode = upstream.status;
    const responseTokens = (upstream.headers.get("connection") || "").toLowerCase().split(",").map(value => value.trim());
    for (const [key, value] of upstream.headers) {
      if (HOP_HEADERS.has(key) || responseTokens.includes(key) || key === "content-encoding" || key === "set-cookie") continue;
      if (key === "location") {
        const location = new URL(value, url);
        res.setHeader(key, location.origin === url.origin ? location.pathname + location.search + location.hash : value);
      } else {
        res.setHeader(key, value);
      }
    }
    const cookies = upstream.headers.getSetCookie();
    if (cookies.length) res.setHeader("Set-Cookie", cookies);
    if (!res.hasHeader("Cache-Control")) res.setHeader("Cache-Control", "no-store");
    if (req.method === "HEAD" || !upstream.body) { res.end(); return; }
    await pipeline(Readable.fromWeb(upstream.body), res);
  } catch (error) {
    if (!res.headersSent && !res.destroyed && !controller.signal.aborted) {
      console.error("Internal API request failed:", error instanceof Error ? error.message : "Unknown error");
      jsonError(res, 502, "The experiment API could not be reached. Please try again.");
    } else if (!res.destroyed) {
      res.destroy();
    }
  } finally {
    req.off("aborted", abort);
    res.off("close", disconnected);
  }
}
