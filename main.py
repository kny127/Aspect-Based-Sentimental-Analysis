# main.py - Simplified Version without Focal Loss and Threshold Optimization

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW 
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import copy

current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

from dataset import ABSAMRCDataset, TRAIN_FILE, VAL_FILE, TEST_FILE, tokenizer, \
                     entity_property_pair, ce_name_to_id, ce_id_to_name, \
                     polarity_name_to_id, polarity_id_to_name, id_to_sentiment, MODEL_NAME
from model import ABSAMRCJointModel

# Hyperparameters 
MAX_LENGTH = 256
BATCH_SIZE = 16  
LEARNING_RATE = 2e-5  # Standard RoBERTa learning rate
EPS = 1e-8
NUM_TRAIN_EPOCHS = 30  
CLASSIFIER_HIDDEN_SIZE = 768
CLASSIFIER_DROPOUT_PROB = 0.1  
MAX_GRAD_NORM = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(model, train_loader, val_loader, device, num_epochs, learning_rate, eps, weight_decay=0.01):

    bert_params = list(model.transformer.parameters())
    other_params = [p for p in model.parameters() if not any(p is bp for bp in bert_params)]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': learning_rate},
        {'params': other_params, 'lr': learning_rate * 10}  
    ], eps=eps, weight_decay=weight_decay)

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, 
        num_warmup_steps=num_training_steps // 10,  
        num_training_steps=num_training_steps
    )

    model.to(device)
    best_val_f1_pipeline = -1
    patience = 7  # Early stopping patience
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        ce_total_loss = 0
        polarity_total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            ce_labels = batch['ce_label'].to(device)
            polarity_labels = batch['polarity_label'].to(device)

            optimizer.zero_grad()

            # Forward pass with loss calculation
            outputs = model(input_ids, attention_mask, adj_matrix, 
                          return_loss=True, ce_labels=ce_labels, polarity_labels=polarity_labels)
            
            total_loss = outputs['loss']
            loss_ce = outputs['ce_loss']
            loss_polarity = outputs['polarity_loss']
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()
            
            total_train_loss += total_loss.item()
            ce_total_loss += loss_ce.item()
            polarity_total_loss += loss_polarity.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_ce_loss = ce_total_loss / len(train_loader)
        avg_polarity_loss = polarity_total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Train - Total: {avg_train_loss:.4f}, CE: {avg_ce_loss:.4f}, Polarity: {avg_polarity_loss:.4f}")

        # Validation
        val_results = evaluate(model, val_loader, device)
        current_pipeline_f1 = val_results['unified_f1_results']['entire pipeline result']['F1']
        ce_f1 = val_results['unified_f1_results']['category extraction result']['F1']
        
        print(f"Epoch {epoch+1} Validation - CE Acc: {val_results['ce_accuracy']:.4f}, "
              f"CE F1: {ce_f1:.4f}, Pipeline F1: {current_pipeline_f1:.4f}")

        if current_pipeline_f1 > best_val_f1_pipeline:
            best_val_f1_pipeline = current_pipeline_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'f1_score': best_val_f1_pipeline,
                'ce_f1': ce_f1
            }, "best_absa_mrc_model.pt")
            print(f"✅ Saved best model: Pipeline F1 = {best_val_f1_pipeline:.4f}, CE F1 = {ce_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break

def evaluate(model, data_loader, device, debug=False):
    model.eval()
    
    ce_true_labels = []
    ce_predictions = []
    polarity_true_labels = []
    polarity_predictions = []
    
    true_data_for_f1 = {}
    pred_data_for_f1 = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            ce_labels = batch['ce_label'].to(device)
            polarity_labels = batch['polarity_label'].to(device)
            
            item_ids = batch['id'] 
            sentences = batch['sentence_form']
            pairs = batch['pair']
            
    
            ce_preds_batch, polarity_preds_batch = model.predict(input_ids, attention_mask, adj_matrix)

            ce_true_labels.extend(ce_labels.cpu().numpy())
            ce_predictions.extend(ce_preds_batch.cpu().numpy())
            
            valid_polarity_mask = (ce_labels == ce_name_to_id['True']).cpu().numpy()
            polarity_true_labels.extend(polarity_labels.cpu().numpy()[valid_polarity_mask])
            polarity_predictions.extend(polarity_preds_batch.cpu().numpy()[valid_polarity_mask])

            # F1 calculation data
            for i in range(input_ids.shape[0]):
                sentence_id = item_ids[i]
                original_sentence_form = sentences[i]
                current_pair = pairs[i]
                
                true_ce_label = ce_labels[i].item()
                true_polarity_label = polarity_labels[i].item()
                
                predicted_ce = ce_preds_batch[i].item()
                predicted_polarity_name = polarity_id_to_name[polarity_preds_batch[i].item()]

                # Initialize data structures
                if sentence_id not in true_data_for_f1:
                    true_data_for_f1[sentence_id] = {
                        "id": sentence_id,
                        "sentence_form": original_sentence_form,
                        "annotation": []
                    }
                
                if sentence_id not in pred_data_for_f1:
                    pred_data_for_f1[sentence_id] = {
                        "id": sentence_id,
                        "sentence_form": original_sentence_form,
                        "annotation": []
                    }
                
                # Add true annotations
                if true_ce_label == ce_name_to_id['True'] and true_polarity_label != -100:
                    true_polarity_name = id_to_sentiment[true_polarity_label]
                    true_data_for_f1[sentence_id]["annotation"].append([current_pair, [None, 0, 0], true_polarity_name])

                # Add predicted annotations
                if predicted_ce == 1:  # CE True
                    pred_data_for_f1[sentence_id]["annotation"].append([current_pair, predicted_polarity_name])

    true_data_list = list(true_data_for_f1.values())
    pred_data_list = list(pred_data_for_f1.values())

    # Calculate metrics
    ce_accuracy = accuracy_score(ce_true_labels, ce_predictions)
    valid_polarity_true = [label for label in polarity_true_labels if label != -100]
    valid_polarity_pred = [pred for i, pred in enumerate(polarity_predictions) if polarity_true_labels[i] != -100]
    polarity_accuracy = accuracy_score(valid_polarity_true, valid_polarity_pred) if len(valid_polarity_true) > 0 else 0.0

    unified_f1_results = evaluation_f1(true_data_list, pred_data_list)
    
    return {
        'ce_accuracy': ce_accuracy,
        'polarity_accuracy': polarity_accuracy,
        'unified_f1_results': unified_f1_results
    }

def predict(model, test_loader, device):
    model.eval()
    model.to(device)
    
    predictions_grouped_by_id = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting on Test Data"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)
            item_ids = batch['id']
            sentences = batch['sentence_form']
            pairs = batch['pair']

            ce_preds, polarity_preds = model.predict(input_ids, attention_mask, adj_matrix)

            for i in range(input_ids.shape[0]):
                sentence_id = item_ids[i]
                original_sentence_form = sentences[i]
                current_pair = pairs[i]
                predicted_ce = ce_preds[i].item()
                predicted_polarity = polarity_id_to_name[polarity_preds[i].item()]

                if sentence_id not in predictions_grouped_by_id:
                    predictions_grouped_by_id[sentence_id] = {
                        "id": sentence_id,
                        "sentence_form": original_sentence_form,
                        "annotation": []
                    }
                
                if predicted_ce == 1:  # CE True
                    predictions_grouped_by_id[sentence_id]["annotation"].append([current_pair, predicted_polarity])

    return list(predictions_grouped_by_id.values())

def evaluation_f1(true_data_list, pred_data_list):
    """F1 evaluation function"""
    ce_eval = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    pipeline_eval = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    pred_data_map = {item['id']: item for item in pred_data_list}

    for true_item in true_data_list:
        true_id = true_item['id']
        pred_item = pred_data_map.get(true_id)

        if pred_item is None:
            for y_ano in true_item['annotation']:
                y_polarity = y_ano[2]
                if y_polarity != '------------':
                    ce_eval['FN'] += 1
                    pipeline_eval['FN'] += 1
            continue

        for y_ano in true_item['annotation']:
            y_category = y_ano[0]
            y_polarity = y_ano[2]

            if y_polarity == '------------':
                continue

            is_ce_found_in_pred = False
            is_pipeline_found_in_pred = False

            for p_ano in pred_item['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found_in_pred = True
                    if y_polarity == p_polarity:
                        is_pipeline_found_in_pred = True
                    break
            
            if is_ce_found_in_pred:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found_in_pred:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

        for p_ano in pred_item['annotation']:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            is_ce_found_in_true = False
            is_pipeline_found_in_true = False

            for y_ano in true_item['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_polarity == '------------':
                    continue

                if y_category == p_category:
                    is_ce_found_in_true = True
                    if y_polarity == p_polarity:
                        is_pipeline_found_in_true = True
                    break

            if not is_ce_found_in_true:
                ce_eval['FP'] += 1
            
            if not is_pipeline_found_in_true:
                pipeline_eval['FP'] += 1

    # Calculate F1 scores
    ce_precision = ce_eval['TP'] / (ce_eval['TP'] + ce_eval['FP']) if (ce_eval['TP'] + ce_eval['FP']) > 0 else 0.0
    ce_recall = ce_eval['TP'] / (ce_eval['TP'] + ce_eval['FN']) if (ce_eval['TP'] + ce_eval['FN']) > 0 else 0.0
    ce_f1 = 2 * ce_recall * ce_precision / (ce_recall + ce_precision) if (ce_recall + ce_precision) > 0 else 0.0

    pipeline_precision = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FP']) if (pipeline_eval['TP'] + pipeline_eval['FP']) > 0 else 0.0
    pipeline_recall = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FN']) if (pipeline_eval['TP'] + pipeline_eval['FN']) > 0 else 0.0
    pipeline_f1 = 2 * pipeline_recall * pipeline_precision / (pipeline_recall + pipeline_precision) if (pipeline_recall + pipeline_precision) > 0 else 0.0

    return {
        'category extraction result': {
            'Precision': ce_precision,
            'Recall': ce_recall,
            'F1': ce_f1
        },
        'entire pipeline result': {
            'Precision': pipeline_precision,
            'Recall': pipeline_recall,
            'F1': pipeline_f1
        }
    }

if __name__ == "__main__":
    print("Loading datasets...")
    train_dataset = ABSAMRCDataset(TRAIN_FILE, tokenizer, entity_property_pair, ce_name_to_id, polarity_name_to_id, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = ABSAMRCDataset(VAL_FILE, tokenizer, entity_property_pair, ce_name_to_id, polarity_name_to_id, max_length=MAX_LENGTH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = ABSAMRCDataset(TEST_FILE, tokenizer, entity_property_pair, ce_name_to_id, polarity_name_to_id, max_length=MAX_LENGTH, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    num_ce_labels = len(ce_name_to_id)
    num_polarity_labels = len(polarity_name_to_id)
    model = ABSAMRCJointModel(
        MODEL_NAME, len(tokenizer), num_ce_labels, num_polarity_labels, 
        CLASSIFIER_HIDDEN_SIZE, CLASSIFIER_DROPOUT_PROB
    )
    
    print(f"\n🚀 Starting training with klue/roberta-base:")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"   - Strategy: No class weights - let model learn naturally")
    print(f"   - Equal loss weighting: CE + Polarity")
    
    train(model, train_loader, val_loader, device, 
          num_epochs=NUM_TRAIN_EPOCHS, learning_rate=LEARNING_RATE, eps=EPS)

    print("\nLoading best model for final evaluation and prediction...")
    checkpoint = torch.load("best_absa_mrc_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    final_val_results = evaluate(model, val_loader, device, debug=True)
    print(f"\n📊 Final Validation Results:")
    print(f"   CE F1: {final_val_results['unified_f1_results']['category extraction result']['F1']:.4f}")
    print(f"   Pipeline F1: {final_val_results['unified_f1_results']['entire pipeline result']['F1']:.4f}")
    
    # Generate test predictions
    predictions = predict(model, test_loader, device)
    
    # Save predictions
    output_file_path = "submission.jsonl"
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in predictions:
            formatted_annotations = []
            for ann in item['annotation']:
                category = ann[0]
                polarity = ann[1]
                formatted_annotations.append([category, polarity])
            
            output_item = {
                "id": item["id"],
                "sentence_form": item["sentence_form"],
                "annotation": formatted_annotations
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    print(f"\n✅ Predictions saved to {output_file_path}")
    print(f"📈 Best validation F1 score: {checkpoint['f1_score']:.4f}")