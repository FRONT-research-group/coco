import json
from confluent_kafka import Consumer

consumer = Consumer({
    "bootstrap.servers": "localhost:29092",  # external access
    "group.id": "tf-privacy",
    "auto.offset.reset": "earliest"
})

consumer.subscribe(["trust-scores"])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"[Kafka Error] {msg.error()}")
        continue

    payload = json.loads(msg.value().decode("utf-8"))
    print(payload)
