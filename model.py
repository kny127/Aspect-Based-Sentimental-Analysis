# model.py - Simplified Version without Focal Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class SimpleClassifier(nn.Module):

    def __init__(self, hidden_size, num_label, dropout_prob):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense3 = nn.Linear(hidden_size // 2, num_label)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()  # GELU instead of Tanh

    def forward(self, features):
        x = features
        
        # First layer
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.dense3(x)
        return x

class AttentionPooling(nn.Module):
    """
    Attention-based pooling for better feature extraction
    """
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, sequence_output, attention_mask):
        
        attention_weights = self.attention(sequence_output) 
        attention_weights = attention_weights.squeeze(-1)    
        

        attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=-1)  

        pooled_output = torch.bmm(attention_weights.unsqueeze(1), sequence_output)  
        pooled_output = pooled_output.squeeze(1)  
        
        return pooled_output

class ImprovedGCNLayer(nn.Module):
    """
    Enhanced GCN with edge features and attention
    """
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(ImprovedGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-head attention for graph
        self.num_heads = 8
        self.head_dim = output_dim // self.num_heads
        
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Residual connection
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, features, adj):
        batch_size, seq_len, _ = features.size()
        residual = features
        
        # Multi-head attention
        Q = self.query(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)  
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply adjacency mask
        adj_expanded = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(adj_expanded == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        if residual.size() == context.size():
            output = context + residual
        else:
            output = context
        
        output = self.layer_norm(output)
        
        return output

class ABSAMRCJointModel(nn.Module):
    def __init__(self, model_name, len_tokenizer, num_ce_labels, num_polarity_labels, 
                 classifier_hidden_size=768, classifier_dropout_prob=0.1):
        super(ABSAMRCJointModel, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        self.transformer.resize_token_embeddings(len_tokenizer)

        bert_hidden_size = self.config.hidden_size

        # GCN Layer
        self.gcn = ImprovedGCNLayer(bert_hidden_size, bert_hidden_size, classifier_dropout_prob)
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(bert_hidden_size)
        
        # Feature fusion
        fusion_input_size = bert_hidden_size * 2
        self.feature_fusion = nn.Linear(fusion_input_size, bert_hidden_size)
        self.fusion_dropout = nn.Dropout(classifier_dropout_prob)
        
        # Classifiers
        self.ce_classifier = SimpleClassifier(bert_hidden_size, num_ce_labels, classifier_dropout_prob)
        self.polarity_classifier = SimpleClassifier(bert_hidden_size, num_polarity_labels, classifier_dropout_prob)
        
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.polarity_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, adj_matrix, return_loss=False, 
                ce_labels=None, polarity_labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs.last_hidden_state

        # GCN processing
        gcn_output = self.gcn(sequence_output, adj_matrix)

        # Attention pooling for both BERT and GCN features
        bert_pooled = self.attention_pooling(sequence_output, attention_mask)
        gcn_pooled = self.attention_pooling(gcn_output, attention_mask)
        
        # Combine features
        combined_features = torch.cat([bert_pooled, gcn_pooled], dim=-1)

        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        fused_features = F.relu(fused_features)
        fused_features = self.fusion_dropout(fused_features)

        # Predictions
        ce_logits = self.ce_classifier(fused_features)
        polarity_logits = self.polarity_classifier(fused_features)

        if return_loss and ce_labels is not None:
            # Calculate losses
            loss_ce = self.ce_loss_fn(ce_logits, ce_labels)
            
            # Polarity loss only for samples with CE = True
            valid_polarity_mask = (ce_labels == 1)  # True class
            if torch.any(valid_polarity_mask):
                masked_polarity_logits = polarity_logits[valid_polarity_mask]
                masked_polarity_labels = polarity_labels[valid_polarity_mask]
                loss_polarity = self.polarity_loss_fn(masked_polarity_logits, masked_polarity_labels)
            else:
                loss_polarity = torch.tensor(0.0, device=ce_logits.device)
            
            # Equal weighting of CE and polarity losses
            total_loss = loss_ce + loss_polarity
            
            return {
                'loss': total_loss,
                'ce_loss': loss_ce,
                'polarity_loss': loss_polarity,
                'ce_logits': ce_logits,
                'polarity_logits': polarity_logits
            }
        
        return ce_logits, polarity_logits

    def predict(self, input_ids, attention_mask, adj_matrix):
        """
        Standard prediction without threshold adjustment
        """
        self.eval()
        with torch.no_grad():
            ce_logits, polarity_logits = self.forward(input_ids, attention_mask, adj_matrix)

            ce_preds = torch.argmax(ce_logits, dim=-1)
            polarity_preds = torch.argmax(polarity_logits, dim=-1)
            
            return ce_preds, polarity_preds