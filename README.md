# coco
Deployment and demo-ready SAFE-6G Cognitive coordinator.

Initial version in this repo: https://github.com/FRONT-research-group/Cognitive_Coordinator

Credits: [Ilias](https://github.com/IliasAlex) 


SOS: A .env file with the model directory (only the directory). Save the model to a path/to/dir/bert_model.pth


```bash
curl -X POST http://localhost:8000/data/submit \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      { "label": "Privacy", "text": "Users should control access to their data." },
      { "label": "Privacy", "text": "Data collection should be transparent and consent-based." },
      { "label": "Privacy", "text": "Personal data must not be shared without authorization." },
      { "label": "Reliability", "text": "The system should be up during peak hours." },
      { "label": "Reliability", "text": "Downtime should not exceed SLA-defined thresholds." },
      { "label": "Reliability", "text": "Service availability must be 99.999%." },
      { "label": "Security", "text": "All communications must be encrypted." },
      { "label": "Security", "text": "Access should be restricted via multi-factor authentication." },
      { "label": "Security", "text": "The platform should detect and block intrusion attempts." },
      { "label": "Resilience", "text": "The network should recover from failures." },
      { "label": "Resilience", "text": "Redundancy must be built into critical components." },
      { "label": "Resilience", "text": "The system should continue to operate under attack." },
      { "label": "Safety", "text": "The system must protect against harm." },
      { "label": "Safety", "text": "User-facing failures should fail-safe, not fail-open." },
      { "label": "Safety", "text": "Physical equipment should not pose danger to users." }
    ]
  }'

curl -X POST http://localhost:8000/lotw/calculate
curl http://localhost:8000/lotw/nlotw
```