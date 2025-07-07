import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score

from coco.utils.logger import get_logger

logger = get_logger("BERT_Trainer")

class Trainer:
    """
        Trainer class for training and validating a model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        save_dir: str,
        lr: float=2e-5,
        weight_decay: float=0.01,
        early_stopping_patience: int=5,
        max_grad_norm: float=1.0
    ) -> None:
        """
        Initialize a Trainer instance.

        Args:
            model (nn.Module): the model to be trained
            train_dataloader (DataLoader): the dataloader for training
            val_dataloader (DataLoader): the dataloader for validation
            test_dataloader (DataLoader): the dataloader for testing
            device (torch.device): the device to be used for training
            save_dir (str): the directory to save the model and logs
            lr (float, optional): the learning rate. Defaults to 2e-5.
            weight_decay (float, optional): the weight decay. Defaults to 0.01.
            early_stopping_patience (int, optional): the patience for early stopping. Defaults to 3.
            max_grad_norm (float, optional): the maximum gradient norm. Defaults to 1.0.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience

    def train(self, epochs: int=10) -> None:
        """
        Train the model for a given number of epochs.

        Args:
        - epochs (int, optional): the number of epochs to train. Defaults to 10.

        Returns:
        - None

        Notes:
        - This method trains the model for a given number of epochs.
        - It also performs early stopping based on the validation loss.
        - The model is saved at the best validation loss.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        total_metrics = {}

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            self.model.train()
            running_loss = 0.0
            all_targets = []
            all_predictions = []

            for batch in tqdm(self.train_dataloader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                score = batch['score'].to(self.device)
                class_type = batch['class_type']

                self.optimizer.zero_grad()

                # Forward pass
                outputs = []
                for i in range(len(input_ids)):
                    output = self.model(
                        input_ids[i].unsqueeze(0),
                        attention_mask[i].unsqueeze(0),
                        class_type[i]
                    )
                    outputs.append(output)

                outputs = torch.cat(outputs).squeeze(1)
                loss = self.criterion(outputs, score)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                running_loss += loss.item()
                all_predictions.extend(outputs.detach().cpu().numpy())
                all_targets.extend(score.cpu().numpy())

            # Training metrics
            mae, mse, rmse, r2, mape = self.calculate_metrics(all_targets, all_predictions)
            avg_train_loss = running_loss / len(self.train_dataloader)
            logger.info(f"Training loss: {avg_train_loss:.4f}")
            logger.info(f"Training Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")

            # Validation
            metrics = self.validate()

            val_loss = metrics['val_loss']

            metrics["train_loss"] = avg_train_loss

            total_metrics[epoch] = metrics

            # Early stopping check
            if val_loss < best_val_loss:
                logger.info("Validation loss improved. Saving model.")
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{self.early_stopping_patience}")
                if patience_counter >= self.early_stopping_patience:
                    logger.info("⏹Early stopping triggered.")
                    break

        test_metrics = self.validate(mode="test")

        total_metrics = {
            "train": total_metrics,
            "test": test_metrics
        }

        # Save metrics
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(total_metrics, f, indent=4)

        return total_metrics

    def validate(self, mode: str="val") -> float:
        """
        Validates the model on the validation dataset and calculates validation loss and metrics.

        This function sets the model to evaluation mode and iterates over the validation dataloader
        without computing gradients. It calculates the validation loss using the specified criterion
        and aggregates prediction and target values to compute evaluation metrics.

        Returns:
            float: The average validation loss over the entire validation dataset.

        Side Effects:
            - Prints the validation loss and various evaluation metrics.
            - Updates the attribute `self.final_val_loss` with the average validation loss.
        """
        if mode == "test":
            logger.info("Validating on test dataset")

            # Load the best model
            best_model_path = os.path.join(self.save_dir, "best_model.pth")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            logger.info("Validating on validation dataset")
            self.model.eval()

        running_val_loss = 0.0
        all_targets = []
        all_predictions = []

        if mode == "val":
            loader = self.val_dataloader
        elif mode == "test":
            loader = self.test_dataloader

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                score = batch['score'].to(self.device)
                class_type = batch['class_type']

                outputs = []
                for i in range(len(input_ids)):
                    output = self.model(
                        input_ids[i].unsqueeze(0),
                        attention_mask[i].unsqueeze(0),
                        class_type[i]
                    )
                    outputs.append(output)

                outputs = torch.cat(outputs).squeeze(1)
                val_loss = self.criterion(outputs, score)
                running_val_loss += val_loss.item()

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(score.cpu().numpy())

        mae, mse, rmse, r2, mape = self.calculate_metrics(all_targets, all_predictions)
        avg_val_loss = running_val_loss / len(loader)

        if mode == "val":
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")
        elif mode == "test":
            logger.info(f"Test loss: {avg_val_loss:.4f}")
            logger.info(f"Test Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")

        self.final_val_loss = avg_val_loss

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "val_loss": avg_val_loss
        }

        return metrics

    def save_model(self) -> None:
        """
        Saves the model with the best validation loss in the specified save directory.

        Notes:
        - The model is saved with the name "best_model.pth".
        """
        save_path = os.path.join(self.save_dir, f"best_model.pth")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved at {save_path}")

    def get_validation_loss(self) -> float:
        """
        Returns the best validation loss found during training.

        Returns:
        - float: The best validation loss. If no validation loss was found, returns None.
        """
        return getattr(self, "final_val_loss", None)
    
    @staticmethod
    def calculate_metrics(targets: list, predictions: list) -> tuple:
        """
        Calculates various evaluation metrics from a list of targets and predictions.

        Args:
        - targets (list): The target values.
        - predictions (list): The predicted values.

        Returns:
        - tuple: The mean absolute error, mean squared error, root mean squared error,
          R-squared, and mean absolute percentage error.
        """
        mae = mean_absolute_error(targets, predictions)
        mse = ((torch.tensor(targets) - torch.tensor(predictions)) ** 2).mean().item()
        rmse = mse ** 0.5
        r2 = r2_score(targets, predictions)
        mape = torch.mean(
            torch.abs((torch.tensor(targets) - torch.tensor(predictions)) / torch.tensor(targets))
        ).item() * 100
        return mae, mse, rmse, r2, mape
