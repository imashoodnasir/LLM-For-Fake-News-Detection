import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Custom Dataset Class
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = "real" if self.labels[index] == 0 else "fake"  # Convert labels to text format
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            label,
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0)
        }

# Preprocessing and Data Loading
def preprocess_and_load_data(file_path, max_len, tokenizer):
    # Load data
    data = pd.read_csv(file_path)
    texts = data['text'].tolist()
    labels = data['label'].tolist()

    # Create Dataset
    dataset = FakeNewsDataset(texts, labels, tokenizer, max_len)
    return dataset

# Fine-tuning T5
def train_t5_model(model, train_loader, val_loader, device, epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Validation
        evaluate_t5_model(model, val_loader, device, tokenizer)

    return model

# Evaluate T5 Model
def evaluate_t5_model(model, data_loader, device, tokenizer):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
            preds = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
            true = [tokenizer.decode(t, skip_special_tokens=True) for t in labels]

            predictions.extend(preds)
            true_labels.extend(true)

    # Convert textual labels back to numeric
    pred_numeric = [0 if p == "real" else 1 for p in predictions]
    true_numeric = [0 if t == "real" else 1 for t in true_labels]

    acc = accuracy_score(true_numeric, pred_numeric)
    precision, recall, f1, _ = precision_recall_fscore_support(true_numeric, pred_numeric, average="weighted")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Main Function
def main():
    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Load dataset
    file_path = "formatted_data.csv"  # Replace with your data file
    dataset = preprocess_and_load_data(file_path, MAX_LEN, tokenizer)

    # Train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_t5_model(model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE)

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_t5")
    tokenizer.save_pretrained("./fine_tuned_t5")

    print("T5 model fine-tuned and saved!")

if __name__ == "__main__":
    main()
