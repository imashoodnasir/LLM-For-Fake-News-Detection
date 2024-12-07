from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Evaluate the model
def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Example usage
evaluate_model(model, val_loader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))