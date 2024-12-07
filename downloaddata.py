import json
import pandas as pd

def extract_text_and_labels(data_path):
    # Load the content and labels
    with open(f"{data_path}/news_content.json", "r") as content_file, \
         open(f"{data_path}/news_labels.json", "r") as labels_file:
        content = json.load(content_file)
        labels = json.load(labels_file)

    # Combine text and labels into a DataFrame
    data = []
    for news_id, text in content.items():
        label = labels.get(news_id)
        if label is not None:  # Ensure matching content and label
            data.append({'text': text, 'label': label})
    
    return pd.DataFrame(data)

# Extract and save GossipCop data
gossipcop_data = extract_text_and_labels('dataset/gossipcop')
gossipcop_data.to_csv('gossipcop_data.csv', index=False)