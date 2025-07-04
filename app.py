# app.py
from flask import Flask, request, jsonify
import pandas as pd
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from flask_cors import CORS
import traceback
from analyze import analyze_url_playwright

app = Flask(__name__)
CORS(app)

# -------- CNN Model Definition --------
class BinaryClassifierCNN(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (input_size // 4), 20)
        self.fc2 = nn.Linear(20, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return torch.sigmoid(x)

# -------- Prediction Pipeline --------
def predict(sample_data, model_path, hf_model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_data['combined'] = (
        sample_data['URL'] + ' ' + sample_data['Domain'] + ' ' +
        sample_data['TLD'] + ' ' + sample_data['Title']
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    bert_model = AutoModel.from_pretrained(hf_model_id).to(device)
    bert_model.eval()

    # Tokenize using list comprehension (fixing earlier bug)
    tokenized = [
        tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=512,
            truncation=True, padding='max_length', return_tensors='pt'
        )
        for text in sample_data['combined']
    ]

    input_ids = torch.cat([x['input_ids'] for x in tokenized])
    attention_masks = torch.cat([x['attention_mask'] for x in tokenized])

    dataloader = DataLoader(TensorDataset(input_ids, attention_masks), batch_size=16)
    all_hidden_states = []

    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_mask = [x.to(device) for x in batch]
            outputs = bert_model(batch_input_ids, attention_mask=batch_attention_mask)
            all_hidden_states.append(outputs.last_hidden_state)

    last_hidden_states = torch.cat(all_hidden_states, dim=0)
    features = last_hidden_states[:, 0, :].cpu().numpy()

    # Drop only allowed columns
    drop_cols = ['URL', 'Domain', 'TLD', 'Title', 'combined']
    num_df = sample_data.drop(columns=[col for col in drop_cols if col in sample_data.columns]).values

    if features.shape[0] != num_df.shape[0]:
        raise ValueError("Mismatch in number of samples between BERT and numeric data")

    combined_features = np.concatenate([features, num_df], axis=1)

    # Standardize
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_features)

    # PCA Transformation
    pca = joblib.load('pca_model.pkl')
    reduced_features = pca.transform(combined_scaled)

    # Model Prediction
    input_size = reduced_features.shape[1]
    model = BinaryClassifierCNN(input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    inputs_tensor = torch.tensor(reduced_features, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inputs_tensor)
        preds = (outputs.squeeze() >= 0.5).int().cpu().numpy()

    return preds.tolist()

# -------- API Endpoint --------
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'error': 'Missing URL'}), 400

        features_dict = analyze_url_playwright(url)
        df = pd.DataFrame([features_dict])
        preds = predict(df, "binary_classifier_cnn.pth", "Babatunde1/BErt_finetuned")
        return jsonify({'url': url, 'prediction': preds})
    except Exception as e:
        traceback.print_exc()  # Print full error in console for debugging
        return jsonify({'error': str(e)}), 500

# -------- Run Server --------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
