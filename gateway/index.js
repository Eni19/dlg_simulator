import express from "express";
import proxy from "./proxy.js";

const app = express();
app.disable("x-powered-by");
// No JSON parser: the proxy forwards the original request body and Content-Type.
app.use("/api", proxy);
app.use((_req, res) => res.status(404).json({ detail: "API route not found" }));

export default app;
