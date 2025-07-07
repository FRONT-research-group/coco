import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from coco.config.config import DATASET_DIR
from coco.utils.logger import get_logger
from coco.data.data_handler import load_dataset

logger = get_logger("Baseline_Trainer")

def run():

    file_path = f"{DATASET_DIR}/dataset.csv"

    # Load the dataset from CSV file
    train_dataset, val_dataset, test_dataset = load_dataset(file_path, batch_size=16, augment=True, model_type="baseline")

    label_encoder = LabelEncoder()
    train_dataset["class_encoded"] = label_encoder.fit_transform(train_dataset["Class"])

    val_dataset["class_encoded"] = label_encoder.transform(val_dataset["Class"])
    test_dataset["class_encoded"] = label_encoder.transform(test_dataset["Class"])

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)

    x_train_text = tfidf.fit_transform(train_dataset["Keywords"])
    y_train = train_dataset["Score"]

    x_val_text = tfidf.transform(val_dataset["Keywords"])
    y_val = val_dataset["Score"]

    x_test_text = tfidf.transform(test_dataset["Keywords"])
    y_test = test_dataset["Score"]

    # Concatenate text and class encoded
    x_train = np.column_stack((x_train_text.toarray(), train_dataset["class_encoded"].values.reshape(-1, 1)))
    x_val = np.column_stack((x_val_text.toarray(), val_dataset["class_encoded"].values.reshape(-1, 1)))
    x_test = np.column_stack((x_test_text.toarray(), test_dataset["class_encoded"].values.reshape(-1, 1)))

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Combine train and val
    x_train = np.vstack((x_train, x_val))
    y_train = np.concatenate((y_train, y_val))

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = ((y_test - y_pred) ** 2).mean()
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test))

    logger.info(f"Baseline Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")

if __name__ == "__main__":
    run()