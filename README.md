# Blitz Person Tracking

Privacy-first vending telemetry + live dashboard.

## What this tracks

- `passed_by`: unique people seen
- `looked_at`: unique people who looked toward machine
- `approached`: unique people who reached engagement zone
- `clicked`: machine-reported button presses
- `purchased`: transaction confirmations

## Run visual demo (no hardware)

```bash
pip install -r requirements.txt
python demo_synthetic.py --port 8000
```

Open `http://localhost:8000/`.

## Integrated verification (your two checklist items)

This validates both:
- synthetic end-to-end pipeline behavior
- `config.yaml` + `README.md` privacy/sensor alignment

```bash
python3 verify_integration.py
```

## Run camera pipeline + dashboard

```bash
python telemetry_engine.py --config config.yaml --dashboard --dashboard-port 8000
```

Optional preview window:

```bash
python telemetry_engine.py --config config.yaml --dashboard --show
```

## Outputs

Each run writes to `runs/<timestamp>/`:

- `annotated.mp4`
- `events.jsonl`
- `funnel.csv`
- `summary.json`

## Privacy posture

- Process frames in memory
- Store counters and events, not identities
- Redact rendered output via `privacy.mode` in `config.yaml`
- Keep machine clicks as ground truth through `/api/click` or POS hooks

## Machine and POS event contract

Use `POST /api/events` for production ingestion (`click`, `purchase`).

Required JSON fields:

- `event_type`: `click` or `purchase`
- `machine_id`: machine identifier
- `idempotency_key`: unique per event (prevents double counting)

Optional:

- `timestamp`: ISO-8601 (`2026-04-22T20:15:02Z`)
- `source`: e.g. `machine_ui`, `pos`

Example click:

```bash
curl -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type":"click",
    "machine_id":"berkeley-01",
    "idempotency_key":"click-1745341092",
    "timestamp":"2026-04-22T20:15:02Z",
    "source":"machine_ui"
  }'
```

Example purchase:

```bash
curl -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type":"purchase",
    "machine_id":"berkeley-01",
    "idempotency_key":"txn-9f2d5a",
    "timestamp":"2026-04-22T20:15:07Z",
    "source":"pos"
  }'
```

### API auth (recommended)

Set an API token before starting the engine:

```bash
export MACHINE_API_TOKEN="replace-with-secret"
```

Then clients send:

```bash
-H "Authorization: Bearer replace-with-secret"
```

If `MACHINE_API_TOKEN` is unset, `/api/events` is open for local testing.
