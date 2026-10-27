# Vercel showcase deployment

This configuration is prepared for review; service names and public exposure still require the owner's confirmation. No cloud deployment or new Git push was performed for this change.

## Online showcase and full local laboratory

Vercel activates the restricted profile automatically (`VERCEL=1`). It runs real
centralized reconstruction on CPU, with one or two distinct bundled CIFAR-100 images per batch,
10, 25 or 50 iterations (also 100 or 200 for two images) and fixed seed/loss settings. Three PNG samples are
included in `src/data/showcase_samples.json`; no dataset download is needed.
Full test-set classification and federated/defense computations are disabled.

The backend rejects configurations outside this policy, including direct calls
to the older compute endpoints. Executions use the stream endpoint; a busy
instance returns an in-stream 429 instead of queueing more work. This admission
limit is per instance, not global across Vercel's autoscaled instances.

The frontend offers the allowed presets and explains that CPU/GPU selection,
batches of up to 8 images, defenses, advanced settings and federated learning
are available in the complete GitHub version. The full version needs more compute
and memory; duration depends on hardware, batch size and configuration.
Unavailable features stay visible with calm, readable descriptions.

Install `requirements-full.txt` for the complete local laboratory. On non-Vercel
hosts, `DLG_APP_MODE=showcase` selects the restricted backend and
`VITE_LAB_MODE=showcase` selects the UI. The UI also reads the profile from health.
The frontend service explicitly builds with `npm run build:showcase`, so it starts
in the restricted UI even before the health request. `DLG_APP_MODE=full`
cannot disable the Vercel restriction. Local preview commands are in [README.md](README.md).

## Services and request flow

| Service | Root | Framework | Public route | Calls |
| --- | --- | --- | --- | --- |
| `frontend` | `frontend` | Vite | `/`, assets, and non-API paths | No internal calls; browser uses same-origin `/api/*` |
| `gateway` | `gateway` | Express | `/api/*` | Internal `app` through the `APP_URL` service binding |
| `app` | `.` | FastAPI, entry point `app:app` | None; internal only | No other service |

Request flow: browser → `/api/*` → gateway function → bound `APP_URL` → FastAPI `/api/*`.

The gateway is a separate service because this frontend is a static Vite app. A browser cannot use runtime service bindings. The gateway reads `process.env.APP_URL` inside each invocation, preserves the original `/api` prefix and query, forwards request bodies, streams responses, and propagates status codes. It has no localhost fallback. Do not create `APP_URL` or a `VITE_APP_URL` variable manually in Vercel, `.env`, or a build command; Vercel injects the binding into the caller.

The gateway's public endpoint does not make the FastAPI service directly public. This separation is not application authentication: users can still call the public gateway endpoints. Authentication/rate limiting policy is not added by this deployment migration.

The top-level `/api/(.*)` rewrite is evaluated before the frontend catch-all. FastAPI already defines `/api/health`, `/api/run-attack`, `/api/run-defense`, and `/api/federated`, so no prefix is stripped. The OpenAPI schema is now `/api/openapi.json`, allowing `/api/docs` and `/api/redoc` to work through the same gateway. `api/index.py` is a legacy alias, not the configured service entry point.

## Run all services locally

From the repository root:

```powershell
npx.cmd vercel@latest dev --local --listen 3005
```

This starts the services together and injects binding URLs. Local-only mode does not link or deploy a cloud project. Open `http://localhost:3005` and check `http://localhost:3005/api/health`. Use `vercel dev` without `--local` only when intentionally working with a linked Vercel project and its cloud settings.

Standalone development remains available: start uvicorn on port 8000 and `npm run dev` under `frontend`. Its development-only proxy defaults to `http://127.0.0.1:8000`, optionally overridden with `DLG_DEV_API_URL` in `frontend/.env.local`. This override is not a service binding, and Vercel Services does not rely on it.

## Runtime and build details

- Python 3.12 is selected in `.python-version`, matching the tested local interpreter. Gateway Node 24 matches the tested local Node runtime.
- Build settings belong to individual services. Legacy top-level `builds`, `routes`, and `PYTHONPATH` overrides were replaced.
- `.vercelignore` excludes local virtual environments, generated files, notebooks, tests, and local environment files. The Python function also excludes the frontend and gateway source from its bundle.
- The frontend code explorer imports Python files from the repository's `src` directory during its Vite build. Keep the entire repository available to service builds; do not upload the `frontend` folder alone.
- Linux installs use the official PyTorch CPU index and explicit CPU builds matching the checked release pair. Other platforms retain the existing version constraints. The deploy's `requirements.txt` excludes Flower, flwr-datasets and matplotlib; the full local `requirements-full.txt` adds them. The showcase does not import federated/scenario modules, which are also excluded from the app function bundle. Their source remains available to the separate frontend build's code explorer.
- Each compute function is configured for a 300-second maximum. An optimization or dataset download that exceeds that duration can still time out; the gateway does not create background jobs or extend the backend deadline.

## Dataset storage

`get_data_root()` resolves an explicit caller path first, then `DLG_DATA_DIR`, then Vercel's temporary filesystem (`<temp>/dlg-lab/cifar100`), otherwise the existing local `~/.torch` cache. Centralized, evaluation, and federated torchvision calls share this resolver. Hugging Face uses a temporary `HF_HOME` on Vercel unless explicitly configured.

Temporary caches are not durable and may disappear between instances. The showcase uses the three bundled PNGs and does not need a dataset cache. No model result is precomputed: every reconstruction is calculated from the selected sample. In the full local profile, existing local/Hugging Face cache behavior remains available. User-provided cache paths must point to writable storage.

## Vercel project settings to confirm

Import the repository as one project with root directory `.`. Do not retain dashboard overrides that conflict with the service configuration.

The scientific Python dependency tree can exceed the standard function bundle size. Confirm Fluid compute and Large Functions eligibility before production deployment. New projects are normally enrolled; an existing project may require `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in its cloud environment. This is a platform opt-in, not a binding URL. Actual Linux bundle size, memory use, dataset download time and full attack latency still need a preview build/run in the target Vercel project.

The service names, exposure table, `gateway → app` binding, and 300-second compute limit are review proposals pending owner confirmation.

## Checks

```powershell
node --test frontend/tests/api-proxy.test.js
python -m unittest discover -s tests -v
npm.cmd --prefix frontend run build
```

Proxy integration tests use a local mock upstream and inject a fixture binding only within the test process. They cover missing binding, invocation-time resolution, prefix/query preservation, POST bodies, authorization, upstream errors, redirects, streams, and trusted destination selection. Cache tests do not import PyTorch or download datasets.

The Vercel local check detected all three services and returned a real FastAPI health response through the gateway's generated binding. A cloud preview deployment has not been performed.

## References

- [Services](https://vercel.com/docs/services)
- [Routing](https://vercel.com/docs/services/routing)
- [Runtime bindings](https://vercel.com/docs/services/bindings)
- [Service configuration](https://vercel.com/docs/services/config-reference)
- [Function filesystem](https://vercel.com/docs/functions/runtimes)
- [Large Functions](https://vercel.com/kb/guide/troubleshooting-function-250mb-limit)
