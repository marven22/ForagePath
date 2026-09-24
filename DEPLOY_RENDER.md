# Deploy the pollinator forage API on Render

## What is deployed

Render runs a FastAPI web service against the immutable V40 SQLite release. The build step expands `deploy/pollinator_forage_v40.sqlite3.gz`, verifies both SHA-256 checksums, runs SQLite integrity checks, confirms the expected published release, and verifies that the spatial-context and Census-boundary artifacts match that release before the server starts. The released application supports all 50 U.S. states. It returns exact-grid guidance only where the frozen national catalog contains an evidence-supported local entry. Otherwise it clearly returns statewide evidence guidance rather than presenting a broad fallback as local observation.

The database is read-only at runtime. Render's ephemeral filesystem is therefore acceptable: every deployment reconstructs the same validated snapshot. Do not store farmer feedback, accounts, or other permanent writes in this SQLite file.

## Deploy

1. Push the prepared deployment repository to a private GitHub, GitLab, or Bitbucket repository supported by Render. The deployment `.gitignore` deliberately excludes raw research data, browser-capture profiles, local secrets, and training workspaces.
2. In Render, choose **New > Blueprint** and select the repository.
3. Render reads `render.yaml` and creates the `pollinator-forage-api` web service and `pollinator-forage-ui` static site.
4. Set the API service's `POLLINATOR_ALLOWED_ORIGINS` to the frontend URL, for example `https://pollinator-forage-ui.onrender.com`. Multiple origins are comma-separated.
5. Set the frontend's `VITE_API_BASE_URL` to the public API URL, for example `https://pollinator-forage-api.onrender.com`.
6. Set the API service's `OPENAI_API_KEY` as a secret. Keep `OPENAI_CHAT_MODEL` at `gpt-5.6-luna` unless deliberately testing another model.
7. Deploy and confirm that `/health` reports `ready` and `/v1/chat/status` reports `available: true`.

The OpenAI key is used only by the server-side chat layer and must never be added to the frontend. The chat model explains results and invokes read-only tools; the deterministic backend remains responsible for every plant recommendation. Chat requests use `store=false` and are not written to the project database.

Farmer feedback uses a separate Render PostgreSQL database and never enters the scientific evidence tables automatically.

The Blueprint currently uses Render's free PostgreSQL plan for prototyping. Render free PostgreSQL databases expire after 30 days and have no backups. Upgrade that database before collecting feedback intended for long-term research use.

## Render commands

Build:

```text
pip install -r requirements-app.txt && python scripts/prepare_render_database.py
```

Start:

```text
uvicorn app_backend.api:app --host 0.0.0.0 --port $PORT --workers 1
```

Health check:

```text
/health
```

## Public endpoints

- `GET /`
- `GET /health`
- `GET /docs`
- `GET /v1/states`
- `GET /v1/states/{state_code}/planting-classes`
- `GET /v1/states/{state_code}/cells`
- `GET /v1/states/{state_code}/boundaries`
- `GET /v1/recommendations`
- `GET /v1/chat/status`
- `POST /v1/chat`
- `POST /v1/feedback`
- `POST /v1/outcomes`

Example:

```text
/v1/recommendations?state=PA&planting_class=herbaceous%20strip&latitude=40.0&longitude=-75.3
```

## Publishing a later data release

The offline research pipeline must build and validate the new release first. Then:

1. Build the new SQLite snapshot.
2. Run the database and recommendation validation suites.
3. Compress the snapshot into `deploy/`.
4. Update the release manifest with both checksums and sizes.
5. Update the expected release identifier in the preparation and API tests.
6. Commit the new artifact. Render will deploy it as one atomic application release.

Runtime requests never retrain models or modify scientific evidence. A location with an evidence-supported local catalog entry receives exact-grid guidance. All other supported locations receive explicitly labeled statewide evidence guidance. The service never guesses a local result.

The frontend also uses OpenStreetMap's standard raster tiles with visible attribution. For a larger public deployment, review traffic and move to an appropriate tile provider if usage outgrows the community tile-service policy.

## Frontend development

Run the API on port 8000, then start the interface:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The default development configuration calls `http://127.0.0.1:8000`.
