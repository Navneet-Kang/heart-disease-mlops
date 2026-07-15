# Heart Disease Grafana Dashboard

This bundle is configured for the metrics exposed by `api/app.py`:

- `api_requests_total`
- `api_request_latency_seconds`
- `model_predictions_total`
- Prometheus `up`

## Docker Compose volumes

Add these two mounts to the Grafana service:

```yaml
volumes:
  - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
  - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
```

Copy:

- `provisioning/datasources/prometheus.yml`
- `provisioning/dashboards/dashboards.yml`
- `dashboards/heart-disease-mlops-dashboard.json`

into your project under `monitoring/grafana/`.

Start the stack:

```bash
docker compose down
docker compose up --build -d
```

Open Grafana at http://localhost:3000 and sign in with the default local credentials if unchanged:

- Username: `admin`
- Password: `admin`

The dashboard appears under the **MLOps** folder.

Generate several requests to `/predict` so the panels contain data.
