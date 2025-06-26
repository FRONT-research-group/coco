import torch.nn as nn
from transformers import BertModel

class BERTForQuantification(nn.Module):
    """
    BERT-based model for quantification of trustworthiness.
    """
    def __init__(self, bert_model_name='bert-base-uncased'):
        """
        Initialize the BERT-based model for quantification.

        The model is initialized with the following components:
        - A BERT model with the given name (default: 'bert-base-uncased')
        - Separate linear heads for each trust function (Reliability, Privacy, Security, Resilience, Safety)
        - The output of the BERT model is passed through the corresponding head to predict the score for each trust function.

        Parameters:
        - bert_model_name (str): The name of the BERT model to use (default: 'bert-base-uncased')
        """
        super(BERTForQuantification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Separate heads for each class (Reliability, Privacy, Security)
        self.reliability_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.privacy_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.security_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.resilience_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.safety_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, class_type):
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
