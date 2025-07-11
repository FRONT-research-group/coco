import torch
import torch.nn as nn

from transformers import BertModel

class BERTForQuantification(nn.Module):
    """
    BERT-based model for quantification of trustworthiness.
    """
    def __init__(self, bert_model_name: str = 'bert-base-uncased') -> None:
        """
        Initialize the BERTForQuantification model.
        
        This constructor sets up a BERT model with multiple regression heads for different trust functions.
        
        Parameters:
            bert_model_name (str): The name of the pretrained BERT model to use. Default is 'bert-base-uncased'.
        
        Attributes:
            bert: The BERT model loaded from the pretrained weights.
            reliability_head: A linear layer to predict scores for the Reliability trust function.
            privacy_head: A linear layer to predict scores for the Privacy trust function.
            security_head: A linear layer to predict scores for the Security trust function.
            resilience_head: A linear layer to predict scores for the Resilience trust function.
            safety_head: A linear layer to predict scores for the Safety trust function.
        """

        super(BERTForQuantification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Separate heads for each class (Reliability, Privacy, Security)
        self.reliability_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.privacy_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.security_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.resilience_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.safety_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, class_type: str) -> torch.Tensor:
        """
        Perform a forward pass to predict the score for the given input text.

        Parameters:
        - input_ids (torch.tensor): The input text as a tensor of token IDs
        - attention_mask (torch.tensor): The attention mask as a tensor of 0s and 1s
        - class_type (str): The trust function to predict a score for (one of: Reliability, Privacy, Security, Resilience, Safety)

        Returns:
        - score (torch.tensor): The predicted score as a tensor of shape (batch_size, 1)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Use the correct regression head based on class_type
        if class_type == "Reliability":
            score = self.reliability_head(pooled_output)
        elif class_type == "Privacy":
            score = self.privacy_head(pooled_output)
        elif class_type == "Security":
            score = self.security_head(pooled_output)
        elif class_type == "Resilience":
            score = self.resilience_head(pooled_output)
        elif class_type == "Safety":
            score = self.safety_head(pooled_output)
        else:
            raise ValueError(f"Invalid class_type. Must be one of: Reliability, Privacy, Security.")
        
        return score