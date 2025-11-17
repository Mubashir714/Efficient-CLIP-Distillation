# Efficient CLIP Distillation

A lightweight, distilled version of CLIP designed to reduce model size and inference latency while retaining strong image–text retrieval performance. This project implements end-to-end teacher–student distillation, evaluation, and benchmarking for efficient multimodal representation learning.

---

## 📌 Overview

This repository contains a complete pipeline for distilling a large CLIP model (teacher) into a smaller, faster student model. The student learns to replicate the teacher’s embedding space using contrastive distillation, enabling deployment in resource-constrained settings without major performance loss.

---

## ✨ Features

- CLIP teacher–student distillation pipeline  
- Contrastive embedding alignment for images and texts  
- Recall@1 evaluation for image–text retrieval  
- Model benchmarking utilities (size + inference speed)  
- Tokenizer handling for teacher and student models  
- Modular and extensible code structure  

---

## 🧰 Tech Stack

- **PyTorch**  
- **OpenCLIP**  
- **Vision Transformers / CNN-based student encoder**  
- **CUDA** (optional)  
- **Python 3.9+**

---

## 📐 Architecture

### Teacher Model
- Pretrained CLIP (e.g., ViT-B/32)  
- High-dimensional joint embedding space  

### Student Model
- Lightweight visual encoder  
- Reduced embedding dimension  
- Optimized for speed and memory efficiency  

### Distillation Objective

The student learns to replicate:

- Teacher image embeddings  
- Teacher text embeddings  
- Teacher similarity logits  

**Final Loss = Image Loss + Text Loss + Contrastive Logit Distillation**

---

## 🔧 Methodology

### 1. Data Processing
- Images normalized to CLIP standards  
- Text tokenized using either the teacher or the student tokenizer  
- Batch collator keeps alignment between both models  

### 2. Training Loop
- Teacher is frozen  
- Student receives gradients from:  
  - Embedding alignment loss  
  - Contrastive similarity loss  

### 3. Evaluation
- Compute embeddings for all image–text pairs  
- Build similarity matrix  
- Measure **Recall@1** retrieval accuracy  

### 4. Model Benchmarking
- Compute model size via state dict  
- Run forward passes for average inference latency  

---

## 📊 Results (Sample Format)

| Model   | Size (MB) | Inference Time (sec) 
|---------|-----------|-----------------------
| Teacher | 605.205191| 0.02715606689453125   
| Student | 398.098845| 0.048318469524383546  

---

