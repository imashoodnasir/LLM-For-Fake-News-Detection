import pandas as pd

# Function to format and save data into a CSV file
def format_to_csv(data, output_path):
    """
    Formats the dataset into a CSV file with 'text' and 'label' columns.

    Parameters:
        data (pd.DataFrame): Preprocessed dataset with content and label columns.
        output_path (str): Path to save the formatted CSV file.

    Returns:
        None
    """
    # Rename columns for clarity
    formatted_data = data.rename(columns={'content': 'text', 'label': 'label'})

    # Save to CSV
    formatted_data.to_csv(output_path, index=False)

    print(f"Data formatted and saved to {output_path}")

# Example Usage
file_path = 'preprocessed_data.csv'  # Input from the preprocessing step
output_csv_path = 'formatted_data.csv'

# Load the preprocessed data
preprocessed_data = pd.read_csv(file_path)

# Format and save to CSV
format_to_csv(preprocessed_data, output_csv_path)