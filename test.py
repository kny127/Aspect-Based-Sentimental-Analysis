# test.py - Test Dataset Prediction with Best Model

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add current directory to path for imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

from dataset import ABSAMRCDataset, TEST_FILE, tokenizer, \
                     entity_property_pair, ce_name_to_id, polarity_name_to_id, \
                     polarity_id_to_name, MODEL_NAME
from model import ABSAMRCJointModel

# Hyperparameters (same as main.py)
MAX_LENGTH = 256
BATCH_SIZE = 16
CLASSIFIER_HIDDEN_SIZE = 768
CLASSIFIER_DROPOUT_PROB = 0.1

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def predict_test_data(model, test_loader, device):
    """
    Generate predictions for test dataset
    """
    model.eval()
    model.to(device)
    
    predictions_grouped_by_id = {}
    
    print("Generating predictions on test data...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            item_ids = batch['id']
            sentences = batch['sentence_form']
            pairs = batch['pair']

            # Get predictions using the model's predict method
            ce_preds, polarity_preds = model.predict(input_ids, attention_mask, adj_matrix)

            # Process each sample in the batch
            for i in range(input_ids.shape[0]):
                sentence_id = item_ids[i]
                original_sentence_form = sentences[i]
                current_pair = pairs[i]
                predicted_ce = ce_preds[i].item()
                predicted_polarity = polarity_id_to_name[polarity_preds[i].item()]

                # Initialize prediction structure for this sentence
                if sentence_id not in predictions_grouped_by_id:
                    predictions_grouped_by_id[sentence_id] = {
                        "id": sentence_id,
                        "sentence_form": original_sentence_form,
                        "annotation": []
                    }
                
                # Add annotation only if CE prediction is True (1)
                if predicted_ce == 1:
                    predictions_grouped_by_id[sentence_id]["annotation"].append([current_pair, predicted_polarity])

    return list(predictions_grouped_by_id.values())

def save_predictions(predictions, output_file_path):
    """
    Save predictions to JSONL file in the required format
    """
    print(f"Saving predictions to {output_file_path}...")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in predictions:
            # Format annotations as required
            formatted_annotations = []
            for ann in item['annotation']:
                category = ann[0]  # entity_property_pair
                polarity = ann[1]  # predicted polarity
                formatted_annotations.append([category, polarity])
            
            # Create output item
            output_item = {
                "id": item["id"],
                "sentence_form": item["sentence_form"],
                "annotation": formatted_annotations
            }
            
            # Write to file
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    print(f"✅ Predictions saved successfully!")
    print(f" Total sentences processed: {len(predictions)}")
    
    # Print some statistics
    total_annotations = sum(len(item['annotation']) for item in predictions)
    sentences_with_annotations = sum(1 for item in predictions if len(item['annotation']) > 0)
    
    print(f"Prediction Statistics:")
    print(f"   - Total annotations: {total_annotations}")
    print(f"   - Sentences with annotations: {sentences_with_annotations}")
    print(f"   - Sentences without annotations: {len(predictions) - sentences_with_annotations}")
    print(f"   - Average annotations per sentence: {total_annotations / len(predictions):.2f}")

def load_best_model(model_path="best_absa_mrc_model.pt"):
    """
    Load the best saved model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model file not found: {model_path}")
    
    print(f"Loading best model from {model_path}...")
    
    # Create model architecture
    num_ce_labels = len(ce_name_to_id)
    num_polarity_labels = len(polarity_name_to_id)
    
    model = ABSAMRCJointModel(
        MODEL_NAME, 
        len(tokenizer), 
        num_ce_labels, 
        num_polarity_labels, 
        CLASSIFIER_HIDDEN_SIZE, 
        CLASSIFIER_DROPOUT_PROB
    )
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print model info
    best_f1 = checkpoint.get('f1_score', 'N/A')
    ce_f1 = checkpoint.get('ce_f1', 'N/A')
    
    print(f"✅ Model loaded successfully!")
    print(f" Best validation Pipeline F1: {best_f1}")
    print(f" Best validation CE F1: {ce_f1}")
    
    return model

def main():
    """
    Main function to run test prediction
    """
    print("=" * 60)
    print("🚀 ABSA-MRC Test Prediction Script")
    print("=" * 60)
    
    # Check if test file exists
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    
    print(f" Test file: {TEST_FILE}")
    
    # Load test dataset
    print("\n Loading test dataset...")
    test_dataset = ABSAMRCDataset(
        TEST_FILE, 
        tokenizer, 
        entity_property_pair, 
        ce_name_to_id, 
        polarity_name_to_id, 
        max_length=MAX_LENGTH, 
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    print(f" Test dataset loaded: {len(test_dataset)} samples")
    print(f" Number of batches: {len(test_loader)}")
    
    # Load best model
    print("\n Loading best model...")
    model = load_best_model()
    
    # Generate predictions
    print("\n Generating predictions...")
    predictions = predict_test_data(model, test_loader, device)
    
    # Save predictions
    print("\n Saving predictions...")
    output_file_path = "submission.jsonl"
    save_predictions(predictions, output_file_path)
    
    print("\n" + "=" * 60)
    print(" Test prediction completed successfully!")
    print(f" Output file: {output_file_path}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check that:")
        print("1. best_absa_mrc_model.pt exists in the current directory")
        print("2. test.jsonl exists in the current directory")
        print("3. All required modules (dataset.py, model.py) are available")
        raise