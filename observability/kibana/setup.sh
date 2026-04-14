#!/bin/bash
# Runs in the background inside the Kibana container after startup.
# Creates data views, imports saved searches, and creates dashboards.

KIBANA_URL="http://localhost:5601"
SETUP_DIR="/usr/share/kibana/setup"
H=(-H 'kbn-xsrf: true' -H 'Content-Type: application/json')

log() { echo "[kibana-setup] $*"; }

# Wait for Kibana to be ready
log "Waiting for Kibana..."
for i in $(seq 1 60); do
  curl -sf "$KIBANA_URL/api/status" > /dev/null 2>&1 && break
  sleep 2
done

# Create data views
log "Creating data views..."
curl -sf -X POST "$KIBANA_URL/api/data_views/data_view" "${H[@]}" \
  -d '{"data_view":{"id":"nemo-logs","title":"nemo-logs","name":"NeMo Logs","timeFieldName":"@timestamp"},"override":true}' \
  > /dev/null 2>&1 && log "  nemo-logs: OK" || log "  nemo-logs: FAILED"

curl -sf -X POST "$KIBANA_URL/api/data_views/data_view" "${H[@]}" \
  -d '{"data_view":{"id":"nemo-traces","title":"nemo-traces","name":"NeMo Traces","timeFieldName":"@timestamp","runtimeFieldMap":{"jaeger_url":{"type":"keyword","script":{"source":"if (doc.containsKey(\"trace.id.keyword\") && doc[\"trace.id.keyword\"].size() > 0) { emit(\"http://localhost:16686/trace/\" + doc[\"trace.id.keyword\"].value); } else { emit(\"\"); }"}}}},"override":true}' \
  > /dev/null 2>&1 && log "  nemo-traces: OK" || log "  nemo-traces: FAILED"

curl -sf -X POST "$KIBANA_URL/api/data_views/data_view" "${H[@]}" \
  -d '{"data_view":{"id":"nemo-metrics","title":".ds-metrics-generic.otel-default-*","name":"NeMo Metrics","timeFieldName":"@timestamp"},"override":true}' \
  > /dev/null 2>&1 && log "  nemo-metrics: OK" || log "  nemo-metrics: FAILED"

# Set logs as default
curl -sf -X POST "$KIBANA_URL/api/data_views/default" "${H[@]}" \
  -d '{"data_view_id":"nemo-logs","force":true}' \
  > /dev/null 2>&1

# Import saved searches
log "Importing saved searches..."
curl -sf -X POST "$KIBANA_URL/api/saved_objects/_import?overwrite=true" \
  -H 'kbn-xsrf: true' \
  --form "file=@$SETUP_DIR/saved-objects.ndjson;type=application/x-ndjson" \
  > /dev/null 2>&1 && log "  saved searches: OK" || log "  saved searches: FAILED"

# Create observability dashboard (traces + logs)
# Delete first for idempotency, then create.
log "Creating dashboards..."
curl -sf -X DELETE "$KIBANA_URL/api/saved_objects/dashboard/nemo-observability" \
  -H 'kbn-xsrf: true' > /dev/null 2>&1

curl -sf -X POST "$KIBANA_URL/api/saved_objects/dashboard/nemo-observability" "${H[@]}" \
  -d '{"attributes":{"title":"NeMo Observability","description":"Training traces and logs. Click trace.id to view in Jaeger.","panelsJSON":"[{\"type\":\"search\",\"gridData\":{\"x\":0,\"y\":0,\"w\":24,\"h\":14,\"i\":\"1\"},\"panelIndex\":\"1\",\"title\":\"Recent Traces\",\"embeddableConfig\":{\"savedObjectId\":\"nemo-traces-search\"}},{\"type\":\"search\",\"gridData\":{\"x\":0,\"y\":14,\"w\":24,\"h\":14,\"i\":\"2\"},\"panelIndex\":\"2\",\"title\":\"Key Training Spans\",\"embeddableConfig\":{\"savedObjectId\":\"nemo-slow-spans\"}},{\"type\":\"search\",\"gridData\":{\"x\":0,\"y\":28,\"w\":24,\"h\":14,\"i\":\"3\"},\"panelIndex\":\"3\",\"title\":\"Recent Logs\",\"embeddableConfig\":{\"savedObjectId\":\"nemo-training-logs\"}},{\"type\":\"search\",\"gridData\":{\"x\":0,\"y\":42,\"w\":24,\"h\":10,\"i\":\"4\"},\"panelIndex\":\"4\",\"title\":\"Error Logs\",\"embeddableConfig\":{\"savedObjectId\":\"nemo-error-logs\"}}]","optionsJSON":"{\"useMargins\":true,\"hidePanelTitles\":false}","timeRestore":true,"timeTo":"now","timeFrom":"now-30m","refreshInterval":{"pause":false,"value":5000},"kibanaSavedObjectMeta":{"searchSourceJSON":"{\"query\":{\"query\":\"\",\"language\":\"kuery\"},\"filter\":[]}"}},"references":[{"id":"nemo-traces-search","name":"1:savedObjectId","type":"search"},{"id":"nemo-slow-spans","name":"2:savedObjectId","type":"search"},{"id":"nemo-training-logs","name":"3:savedObjectId","type":"search"},{"id":"nemo-error-logs","name":"4:savedObjectId","type":"search"}]}' \
  > /dev/null 2>&1 && log "  NeMo Observability: OK" || log "  NeMo Observability: FAILED"

log "Done."
