import os
import json
from confluent_kafka import Producer

from coco.app.core.logger import get_logger

logger = get_logger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TRUST_TOPIC = os.getenv("KAFKA_TOPIC", "trust-scores")

producer = Producer({'bootstrap.servers': KAFKA_BROKER})

def publish_trust_scores(nlotw: dict):
    """
    Publish the given trust scores to the Kafka topic specified in the KAFKA_TOPIC environment variable.

    :param nlotw: A dictionary containing the trust scores, where the keys are the labels and the values are the scores.
    :type nlotw: dict
    """
    try:
        message = json.dumps(nlotw)
        producer.produce(TRUST_TOPIC, message.encode("utf-8"))
        producer.flush()
        logger.info(f"Published trust scores to Kafka: {message}")
    except Exception as e:
        logger.error(f"Kafka publish failed: {e}")
