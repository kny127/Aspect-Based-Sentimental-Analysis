import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import os
import pandas as pd
from konlpy.tag import Okt
from collections import defaultdict 

# 전역 변수 
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(current_dir, 'train.jsonl')
VAL_FILE = os.path.join(current_dir, 'validation.jsonl')
TEST_FILE = os.path.join(current_dir, 'test.jsonl')

MODEL_NAME = "klue/roberta-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


okt = Okt()


entity_property_pair = [
    '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
    '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성',
    '패키지/구성품#일반', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
    '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도',
]


ce_id_to_name = ['False', 'True'] 
ce_name_to_id = {ce_id_to_name[i]: i for i in range(len(ce_id_to_name))}


polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}
id_to_sentiment = {v: k for k, v in polarity_name_to_id.items()}


special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}
tokenizer.add_special_tokens(special_tokens_dict)

def get_dependency_adjacency_matrix(sentence, tokenizer, max_length):
    """
    인접 행렬 생성 함수
    """
    tokens = tokenizer.tokenize(sentence)
    num_tokens = min(len(tokens) + 2, max_length)  # +2 for [CLS], [SEP]

    adj_matrix = torch.eye(max_length, max_length, dtype=torch.float32)
    
    # 윈도우 기반 연결 
    window_size = 3  # 각 토큰은 앞뒤 3개의 토큰과 연결
    
    for i in range(1, num_tokens - 1):  # Exclude [CLS] and [SEP]
        for j in range(max(1, i - window_size), min(num_tokens - 1, i + window_size + 1)):
            if i != j:
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  
    
    # [CLS] 토큰과 모든 실제 토큰 연결 
    for i in range(1, num_tokens - 1):
        adj_matrix[0, i] = 1.0
        adj_matrix[i, 0] = 1.0
    
    # [SEP] 토큰과의 연결은 제한적으로만 설정
    if num_tokens > 2:
        adj_matrix[0, num_tokens - 1] = 1.0
        adj_matrix[num_tokens - 1, 0] = 1.0

    return adj_matrix

class ABSAMRCDataset(Dataset):
    def __init__(self, file_path, tokenizer, entity_property_pair, ce_name_to_id, polarity_name_to_id, max_length=256, is_test=False):
        self.tokenizer = tokenizer
        self.entity_property_pair = entity_property_pair
        self.ce_name_to_id = ce_name_to_id
        self.polarity_name_to_id = polarity_name_to_id
        self.max_length = max_length
        self.is_test = is_test
        self.data = self._load_and_process_data(file_path)
        
        # 데이터 통계 출력 (디버깅용)
        if not is_test:
            self._print_data_statistics()

    def _print_data_statistics(self):
        """데이터 통계 정보 출력"""
        ce_true_count = sum(1 for item in self.data if item['ce_label'].item() == 1)
        ce_false_count = len(self.data) - ce_true_count
        
        polarity_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for item in self.data:
            if item['polarity_label'].item() != -100:
                polarity_name = id_to_sentiment[item['polarity_label'].item()]
                polarity_counts[polarity_name] += 1
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total samples: {len(self.data)}")
        print(f"CE - True: {ce_true_count}, False: {ce_false_count}")
        print(f"CE True ratio: {ce_true_count / len(self.data):.3f}")
        print(f"Polarity - Positive: {polarity_counts['positive']}, Negative: {polarity_counts['negative']}, Neutral: {polarity_counts['neutral']}")
        print(f"===========================\n")

    def _load_and_process_data(self, file_path):
        processed_data = []
        raw_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                raw_data.append(json.loads(line))
        
        for item in tqdm(raw_data, desc=f"Processing {file_path}"):
            sentence = item['sentence_form']
            item_id = item['id']
            annotations = item.get('annotation', []) # test 데이터에는 annotation이 없음

            if pd.isna(sentence) or sentence.strip() == "":
                continue

            # Generate adjacency matrix for the sentence
            adj_matrix = get_dependency_adjacency_matrix(sentence, self.tokenizer, self.max_length)

            # 각 entity_property_pair에 대해 하나의 샘플 생성
            for pair in self.entity_property_pair:
                
                combined_text = f"{sentence} [SEP] {pair}"
                
                tokenized_input = self.tokenizer(
                    combined_text,
                    add_special_tokens=True,
                    max_length=self.max_length, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                )

                input_ids = tokenized_input['input_ids'].squeeze()
                attention_mask = tokenized_input['attention_mask'].squeeze()

                if self.is_test:
                    processed_data.append({
                        'id': item_id,
                        'sentence_form': sentence,
                        'pair': pair,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'adj_matrix': adj_matrix
                    })
                else:
                    ce_label = self.ce_name_to_id['False']
                    polarity_label = -100
                    
                    for ann in annotations:
                        ann_category = ann[0]
                        ann_polarity = ann[2]
                        if ann_category == pair:
                            if ann_polarity != '------------':
                                ce_label = self.ce_name_to_id['True']
                                polarity_label = self.polarity_name_to_id[ann_polarity]
                                break
                    
                    processed_data.append({
                        'id': item_id,
                        'sentence_form': sentence,
                        'pair': pair,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'ce_label': torch.tensor(ce_label, dtype=torch.long),
                        'polarity_label': torch.tensor(polarity_label, dtype=torch.long),
                        'adj_matrix': adj_matrix
                    })
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]