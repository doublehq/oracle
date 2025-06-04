import json
from src.service.oracle import Oracle

event = json.load(open("debug.json"))
response = Oracle.process_lambda_event(event)
print(json.dumps(response, indent=2))
