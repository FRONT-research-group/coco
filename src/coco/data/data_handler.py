import random
import pandas as pd
import torch

from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from nltk.corpus import wordnet


class TextDataset(Dataset):
    """
    TextDataset class for handling text data and labels for training and validation.
    """
    def __init__(self, data: list, tokenizer: BertTokenizer, max_len: int=128) -> None:
        """
        Initializes the TextDataset with the given data, tokenizer, and maximum sequence length.

        Args:
            data (list): A list of dictionaries, where each dictionary contains the text data and labels.
            tokenizer (BertTokenizer): The tokenizer to convert text into token ids.
            max_len (int, optional): The maximum length of tokenized sequences. Defaults to 128.
        """

        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves the tokenized representation of the text data and its associated class and score from the dataset.

        Args:
            index (int): The index of the dataset item to be retrieved.

        Returns:
            dict: A dictionary containing:
                - 'input_ids' (torch.Tensor): The token IDs of the text input.
                - 'attention_mask' (torch.Tensor): The attention mask for the tokenized input.
                - 'class_type' (str): The class type of the text (e.g., Reliability, Privacy).
                - 'score' (torch.Tensor): The score associated with the text, as a float tensor.
        """

        sample = self.data[index]
        text = sample['Keywords']  # Text input
        class_type = sample['Class']  # Class type: Reliability, Privacy, etc.
        score = sample['Score']  # The actual score as the target

        # Tokenize the text using the provided tokenizer
        tokens = self.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")

        # Return the tokenized inputs and the corresponding score and class
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'class_type': class_type,
            'score': torch.tensor(score, dtype=torch.float),
        }
    
def synonym_replacement(text: str, n: int=2) -> str:
    """
    Replace n random words in the text with their synonyms.

    Args:
        text (str): The text to be processed.
        n (int, optional): The number of words to be replaced. Defaults to 2.

    Returns:
        str: The processed text with n random words replaced with their synonyms.
    """
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            if synonym != word:  # Ensure synonym is different
                new_words = [synonym if w == word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
    return ' '.join(new_words)

def augment_text(text: str) -> str:
    """
    Apply a random text augmentation technique to the input text.

    Args:
        text (str): The input text to be augmented.

    Returns:
        str: The augmented text after applying a random augmentation technique.
    """
    techniques = [synonym_replacement]
    augmentation = random.choice(techniques)
    return augmentation(text)

def load_data_from_csv(file_path: str, n_augmentations: int=5) -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV file, normalizes the scores between 0 and 1 if needed, and applies data augmentation techniques to the text data.

    Args:
        file_path (str): The path to the CSV file containing the dataset.
        n_augmentations (int, optional): The number of times to apply data augmentation techniques to each sample. Defaults to 3.

    Returns:
        pd.DataFrame: The processed dataset with normalized scores and augmented text data.
    """
    df = pd.read_csv(file_path)
    
    # Normalize scores between 0 and 1 if needed
    df['Score'] = df['Score'].apply(lambda x: sum(map(float, x.split(','))) / 2 / 100)  # Normalize score
    
    # Apply data augmentation
    augmented_rows = []
    for _, row in df.iterrows():
        original_text = row['Keywords']
        for _ in range(n_augmentations):
            augmented_text = augment_text(original_text)
            augmented_rows.append({'Keywords': augmented_text, 'Score': row['Score'], 'Class': row['Class']})
    
    # Append augmented rows to the original dataset
    augmented_df = pd.DataFrame(augmented_rows)
    df = pd.concat([df, augmented_df], ignore_index=True)

    return df

def load_dataset(file_path: str, batch_size: int=16, max_len: int=128, augment: bool=True, split: List[float]=[0.7, 0.15, 0.15], model_type: str='DL') -> tuple[DataLoader, DataLoader]:
    """
    Loads the dataset from a CSV file, normalizes the scores between 0 and 1 if needed, applies data augmentation techniques to the text data, and returns DataLoaders for training and validation.

    Args:
        file_path (str): The path to the CSV file containing the dataset.
        batch_size (int, optional): The batch size for the training and validation DataLoaders. Defaults to 16.
        max_len (int, optional): The maximum length of tokenized sequences. Defaults to 128.
        augment (bool, optional): Whether to apply data augmentation techniques to the text data. Defaults to True.

    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """
    data_df = load_data_from_csv(file_path) if augment else pd.read_csv(file_path)

    train_df, test_df = train_test_split(data_df, test_size=split[1] + split[2], random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=split[2] / (split[1] + split[2]), random_state=42)

    if model_type=="baseline":
        return train_df, val_df, test_df
    else:

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = TextDataset(train_df.to_dict(orient='records'), tokenizer, max_len=max_len)
        val_dataset = TextDataset(val_df.to_dict(orient='records'), tokenizer, max_len=max_len)
        test_dataset = TextDataset(test_df.to_dict(orient='records'), tokenizer, max_len=max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        return train_dataloader, val_dataloader, test_dataloader
    
    #data = df.to_dict(orient='records')

    #print(data)

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #dataset = TextDataset(data, tokenizer, max_len=max_len)

    # Split the dataset into training, validation and test sets: 70, 15 and 15
    #train_size = int(0.7 * len(dataset))
    #val_size = int(0.15 * len(dataset))
    #test_size = len(dataset) - train_size - val_size

    #train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    #return train_dataloader, val_dataloader, test_dataloader