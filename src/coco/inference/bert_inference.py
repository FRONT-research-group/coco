import torch
from transformers import BertTokenizer

from coco.utils.logger import get_logger
from coco.models.bert import BERTForQuantification

logger = get_logger(__name__)

class InferenceHandler():
    """
        Handles inference for the BERT model.
    """
    def __init__(self, model_weights_path):
        """
        Initialize the InferenceHandler class.

        Args:
        - model_weights_path (str): The path to the model weights.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_weights_path)

    def load_model(self, model_weights_path):
        """
        Load the model from the specified path.

        Args:
        - model_weights_path (str): the path to the model weights.

        Returns:
        - model (torch.nn.Module): the loaded model.
        """
        model = BERTForQuantification()
        logger.info(f"Loading model from {model_weights_path}")
        model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def tokenize_input(self, text, tokenizer, max_len=128):
        """
        Tokenize the input text for the model.

        Args:
        - text (str): the input text to tokenize.
        - tokenizer (transformers.PreTrainedTokenizer): the tokenizer to use.
        - max_len (int, optional): the maximum length of the tokenized input. Defaults to 128.

        Returns:
        - input_ids (torch.tensor): the tokenized input IDs.
        - attention_mask (torch.tensor): the attention mask for the tokenized input.
        """
        tokens = tokenizer(text, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt")
        return tokens['input_ids'], tokens['attention_mask']

    def inference(self, text, class_type):
        """
        Perform inference on the input text to predict a score based on the specified class type.

        Args:
        - text (str): The input text for which the score is to be predicted.
        - class_type (str): The type of classification to perform (e.g., "Reliability", "Privacy", etc.).

        Returns:
        - float: The predicted score as a percentage.
        """

        input_ids, attention_mask = self.tokenize_input(text, self.tokenizer)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            score = self.model(input_ids, attention_mask, class_type)
        return score.item() * 100