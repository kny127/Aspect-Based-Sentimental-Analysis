# Aspect-Based-Sentimental-Analysis
This repository contains the implementation of a Dual-Path Model for Aspect-Based Sentiment Analysis (ABSA) on Korean product reviews. The model jointly performs Aspect Category Detection (ACD) and Aspect Sentiment Classification (ASC) using a architecture that combines global contextual features from KLUE/RoBERTa with local context understanding through adjacency-masked attention.

## Overview
The project addresses the task of analyzing sentiment in Korean product reviews at an aspect level. Unlike traditional sentiment analysis that determines overall sentiment, ABSA:   
  + Identifies specific aspects (categories) mentioned in the text
  + Determines sentiment polarity (positive/negative/neutral) for each identified aspect

## Dataset
The model is trained on the 2022 AI Language Proficiency Assessment Competition corpus provided by the National Institute of the Korean Language(국립국어원에서 제공하고 있는 속성 기반 감성 분석 데이터셋 활용
):
 + Training: 2,999 sentences
 + Validation: 2,792 sentences
 + Test: 2,126 sentences
 + Aspects: 21 categories (4 entities × various attributes)
 + Sentiment Classes: Positive, Negative, Neutral

## Performance
The model achieves significant improvements over the baseline:   
|Model|내용|
|------|---|
|Baseline (XLM-RoBERTa)|0.5368|
|Dual-Path Model|0.6666|
|Improvement|+33.72%|

## Model Architecture
### Three-Stage Pipeline:
1. Input Processing Stage
   + Combines sentences with 21 aspect-category pairs
   + Tokenization with special tokens 
   + Adjacency matrix generation for local context
2. Dual-Path Feature Extraction
   + Path 1: KLUE/RoBERTa for global contextual understanding
   + Path 2: Multi-head attention with adjacency masking for local context
3. Feature Fusion & Classification
   + Feature fusion from both paths
   + Parallel classifiers for aspect and sentiment classification
