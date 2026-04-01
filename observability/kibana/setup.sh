#!/bin/bash
# Runs in the background inside the Kibana container after startup.
# Creates the nemo-logs data view and imports saved searches/dashboard.

KIBANA_URL="http://localhost:5601"

# Wait for Kibana to be ready
for i in $(seq 1 60); do
  curl -sf "$KIBANA_URL/api/status" > /dev/null 2>&1 && break
  sleep 2
done

# Create data view
curl -sf -X POST "$KIBANA_URL/api/data_views/data_view" \
  -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{"data_view":{"id":"nemo-logs","title":"nemo-logs","name":"NeMo Logs","timeFieldName":"@timestamp"},"override":true}' \
  > /dev/null 2>&1

# Set as default
curl -sf -X POST "$KIBANA_URL/api/data_views/default" \
  -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{"data_view_id":"nemo-logs","force":true}' \
  > /dev/null 2>&1

# Import saved objects
curl -sf -X POST "$KIBANA_URL/api/saved_objects/_import?overwrite=true" \
  -H 'kbn-xsrf: true' \
  --form file=@/usr/share/kibana/setup/saved-objects.ndjson \
  > /dev/null 2>&1
