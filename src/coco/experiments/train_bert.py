import torch

from coco.data.data_handler import load_dataset
from coco.models.bert import BERTForQuantification
from coco.trainer.train import Trainer

DATASET_DIR = "/home/bilito/Documents/FRONT_RG/coco/notebooks/data"
REGISTRY_DIR = "/home/bilito/Documents/FRONT_RG/coco/notebooks/data/registry"

def run():

    file_path = f"{DATASET_DIR}/dataset.csv"

    # Load the dataset from CSV file
    train_dataloader, val_dataloader = load_dataset(file_path, batch_size=32, augment=True)

    # Initialize the model
    model = BERTForQuantification()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        save_dir=REGISTRY_DIR
    )

    trainer.train(epochs=500)

if __name__ == "__main__":
    run()