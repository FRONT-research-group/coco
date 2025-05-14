import os
import math
from collections import defaultdict
from typing import Dict, List

from coco.app.core import state, config
from coco.app.core.logger import get_logger
from coco.app.models.lotw_models import LabeledText
from coco.app.services.bert_regressor.inference import InferenceHandler

logger = get_logger(__name__)
_inference_handler = InferenceHandler(os.path.join(config.MODEL_STORAGE, "final_bert_model.pth"))

class LoTWService:
    def predict_score(self, text: str, class_type: int) -> float:
        """
        Predicts a score for the given text based on the specified class type.

        Args:
            text (str): The input text for which the score is to be predicted.
            class_type (int): An integer representing the class type (i.e. the trust function) for prediction.

        Returns:
            float: The predicted score as a floating-point value.
        """
        score = _inference_handler.inference(text, class_type)
        logger.info(f"Predicted score for class {class_type}: {score:.2f} (text: {text[:50]}...)")
        return score

    def calculate_nlotw(self, data: List[LabeledText]) -> Dict[str, float]:
        """
        Calculates the normalized LoTW (nLoTW) values for the given labeled data.

        Args:
            data (List[LabeledText]): A list of labeled text data points.

        Returns:
            Dict[str, float]: A dictionary where the keys are the labels and the values are the nLoTW values.
        """
        logger.info(f"Calculating nLoTw for {len(data)} items...")
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item.label].append(item.text)

        logger.info(f"Grouped data by label: {[ (label, len(texts)) for label, texts in grouped_data.items() ]}")

        scores_per_label = {
            label: [self.predict_score(text, label) for text in texts]
            for label, texts in grouped_data.items()
        }

        wtf = self.compute_wtf(scores_per_label)

        nlotw = {
            label: wtf[label] * (sum(scores) / len(scores))
            for label, scores in scores_per_label.items()
        }

        total = sum(nlotw.values())
        if total > 0:
            nlotw = {label: 100 * score / total for label, score in nlotw.items()}

        logger.info(f"nLoTw result: {nlotw}")
        return nlotw

    def compute_wtf(self, scores_per_label: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Computes the weights that fix the trust (wtf) values for each label given the scores per label.

        Args:
            scores_per_label (Dict[str, List[float]]): A dictionary where the keys are the labels and the values are lists of scores.

        Returns:
            Dict[str, float]: A dictionary where the keys are the labels and the values are the wtf values.
        """

        logger.info("Computing W_TF values...")

        wtf = {}
        total_sum = sum(math.log(1 + len(scores)) for scores in scores_per_label.values())

        for label, scores in scores_per_label.items():
            wtf[label] = math.log(1 + len(scores)) / total_sum if total_sum > 0 else 0

        total_wtf = sum(wtf.values())
        if total_wtf > 0:
            wtf = {label: weight / total_wtf for label, weight in wtf.items()}

        logger.info(f"W_TF values: {wtf}")
        return wtf

    def compute_clotw_scores(self) -> Dict[str, float]:
        if not state.stored_data:
            raise ValueError("No data available")

        nlotw = self.calculate_nlotw(state.stored_data)

        state.nlotw_score = nlotw
        state.calculating = False

        return nlotw