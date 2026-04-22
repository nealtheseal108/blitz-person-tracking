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
