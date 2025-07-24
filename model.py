# model.py - Dual-Path Model for ABSA
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class AspectAwareClassifier(nn.Module):
    """
    Multi-layer classifier with batch normalization and GELU activation
    for aspect-based sentiment analysis tasks
    """
    def __init__(self, hidden_size, num_label, dropout_prob):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense3 = nn.Linear(hidden_size // 2, num_label)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()  # GELU activation for better performance

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

class SentenceRepresentation(nn.Module):
    """
    Attention-based mechanism for generating sentence representations
    from token-level features
    """
    def __init__(self, hidden_size):
        super(SentenceRepresentation, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, sequence_output, attention_mask):
        # Calculate attention weights
        attention_weights = self.attention(sequence_output)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.squeeze(-1)    # [batch_size, seq_len]
        
        # Mask padding tokens
        attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        
        # Normalize with softmax
        attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, seq_len]
        
        # Apply attention weights to get sentence representation
        sentence_repr = torch.bmm(attention_weights.unsqueeze(1), sequence_output)  # [batch_size, 1, hidden_size]
        sentence_repr = sentence_repr.squeeze(1)  # [batch_size, hidden_size]
        
        return sentence_repr

class LocalContextAttention(nn.Module):
    """
    Multi-head attention with adjacency matrix masking to focus on local context.
    This implements the dual-path architecture's second pathway.
    """
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(LocalContextAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-head attention configuration
        self.num_heads = 8
        self.head_dim = output_dim // self.num_heads
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Residual connection projection if dimensions differ
        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, features, adj_matrix):
        batch_size, seq_len, _ = features.size()
        residual = features
        
        # Generate Q, K, V
        Q = self.query(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(features).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for multi-head attention: [batch, heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply adjacency matrix masking to focus on local context
        adj_expanded = adj_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores.masked_fill(adj_expanded == 0, -1e9)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Apply residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        output = context + residual
        output = self.layer_norm(output)
        
        return output

class ABSADualPathModel(nn.Module):
    """
    Dual-Path Model for Aspect-Based Sentiment Analysis
    Combines KLUE/RoBERTa contextual embeddings with local context attention
    """
    def __init__(self, model_name, len_tokenizer, num_ce_labels, num_polarity_labels, 
                 classifier_hidden_size=768, classifier_dropout_prob=0.1):
        super(ABSADualPathModel, self).__init__()

        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        self.transformer.resize_token_embeddings(len_tokenizer)

        bert_hidden_size = self.config.hidden_size

        # Local context attention layer (second path)
        self.local_context_attention = LocalContextAttention(
            bert_hidden_size, bert_hidden_size, classifier_dropout_prob
        )
        
        # Sentence representation generator
        self.sentence_representation = SentenceRepresentation(bert_hidden_size)
        
        # Feature fusion layer
        fusion_input_size = bert_hidden_size * 2  # Concatenating two paths
        self.feature_fusion = nn.Linear(fusion_input_size, bert_hidden_size)
        self.fusion_dropout = nn.Dropout(classifier_dropout_prob)
        
        # Task-specific classifiers
        self.ce_classifier = AspectAwareClassifier(
            bert_hidden_size, num_ce_labels, classifier_dropout_prob
        )
        self.polarity_classifier = AspectAwareClassifier(
            bert_hidden_size, num_polarity_labels, classifier_dropout_prob
        )
        
        # Loss functions
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.polarity_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, adj_matrix, return_loss=False, 
                ce_labels=None, polarity_labels=None):
        # Path 1: KLUE/RoBERTa contextual embeddings
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )
        sequence_output = transformer_outputs.last_hidden_state

        # Path 2: Local context attention with adjacency masking
        local_context_output = self.local_context_attention(sequence_output, adj_matrix)

        # Generate sentence representations for both paths
        transformer_repr = self.sentence_representation(sequence_output, attention_mask)
        local_context_repr = self.sentence_representation(local_context_output, attention_mask)
        
        # Feature fusion
        combined_features = torch.cat([transformer_repr, local_context_repr], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = F.relu(fused_features)
        fused_features = self.fusion_dropout(fused_features)

        # Task predictions
        ce_logits = self.ce_classifier(fused_features)
        polarity_logits = self.polarity_classifier(fused_features)

        if return_loss and ce_labels is not None:
            # Calculate CE loss
            loss_ce = self.ce_loss_fn(ce_logits, ce_labels)
            
            # Calculate polarity loss only for samples with CE = True
            valid_polarity_mask = (ce_labels == 1)
            if torch.any(valid_polarity_mask):
                masked_polarity_logits = polarity_logits[valid_polarity_mask]
                masked_polarity_labels = polarity_labels[valid_polarity_mask]
                loss_polarity = self.polarity_loss_fn(masked_polarity_logits, masked_polarity_labels)
            else:
                loss_polarity = torch.tensor(0.0, device=ce_logits.device)
            
            # Equal weighting of losses
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
        Make predictions for aspect presence and sentiment polarity
        """
        self.eval()
        with torch.no_grad():
            ce_logits, polarity_logits = self.forward(input_ids, attention_mask, adj_matrix)

            ce_preds = torch.argmax(ce_logits, dim=-1)
            polarity_preds = torch.argmax(polarity_logits, dim=-1)
            
            return ce_preds, polarity_preds
