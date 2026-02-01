---
marp: true
theme: default
---
# Neural Network-Based Chess Motif Detection

---
# Project Description

Modern chess engines excel at calculating optimal moves but often fail to explain *why* those moves are strong in terms that humans understand. This project aims to bridge that gap by building a neural network capable of detecting tactical and positional motifs in chess positions.

Using a large dataset of annotated chess puzzles, the system will analyze board states and output the motifs present in each position as a multi-label classification problem. The tool is designed primarily as a research project, focusing on neural network design, dataset engineering, and motif detection accuracy rather than production deployment.

The end goal is to create a fast, interpretable system that enhances chess learning by identifying patterns such as forks, pins, and discovered attacks.

---
# Features and Requirements

## Core Features

### Motif Detection
**Requirements:**
- Detect well-established chess motifs (e.g., pins, forks, discovered attacks, weak squares, open files).
- Accept any valid FEN position as input.
- Achieve performance significantly better than random classification.
- Support expansion as additional motifs are introduced.

### Multi-label Output
**Requirements:**
- Allow multiple motifs per position.
- Represent outputs as a binary vector (~130 motifs).
- Use sigmoid activation for independent motif probabilities.
- Evaluate using multi-label metrics such as F1 score, precision, and recall.

### Engine Analysis Integration (Optional)
**Requirements:**
- Provide engine evaluations for positions.
- Optionally display best moves alongside detected motifs.
- Use engine output as a validation mechanism for predictions.

### Readable UI
**Requirements:**
- Display chess boards in ASCII/terminal format.
- Output detected motifs in a clear list.
- Maintain fast response time for near real-time analysis.

---
# Data Model and Architecture

## Dataset
- Source: Lichess puzzle dataset (via Kaggle).
- File size: ~460MB CSV containing hundreds of thousands of puzzles, with millions available.
- Sprint 1 subset: **10,000 puzzles** for proof-of-concept.

**Key Fields Used:**
- FEN (board state)
- Themes (motif labels)
- Moves (optional auxiliary data)
- Rating / Popularity (optional filtering for data quality)

## Data Pipeline
1. Sample 10,000 representative puzzles.
2. Convert FEN strings into 8×8×12 tensors (piece-type channels).
3. Map puzzle themes to a standardized motif set (~130 labels).
4. Generate multi-label binary vectors.
5. Split dataset into training (80%), validation (10%), and test (10%).
6. Store processed tensors for efficient training.

## Neural Network Architecture (Initial)

**Type:** Convolutional Neural Network (CNN)

**Approximate Size:** ~50k–200k parameters  
(Small enough for fast laptop GPU training)

**Structure:**
- Input: 8×8×12 tensor
- 2–4 convolutional layers (32–64 filters, 3×3 kernels)
- Minimal pooling to preserve spatial relationships
- 1–2 dense layers (128–256 neurons)
- Output layer: ~130 sigmoid neurons (multi-label)

**Training Hardware:**
- Laptop GPU (4GB GDDR6, Ampere architecture)
- Expected fast training and millisecond-level inference.

**Design Goal:**  
Start small for rapid experimentation, then scale dataset and model complexity in Sprint 2.

---
# Tests

## Model Evaluation
- Multi-label F1 score
- Precision and recall per motif
- Validation vs. training loss to detect overfitting

## Functional Tests
- Verify correct FEN-to-tensor conversion.
- Confirm accurate motif vector generation.
- Ensure inference runs within milliseconds.
- Validate terminal output readability.

## Experimentation
- Compare performance across architectures.
- Analyze rare motif prediction accuracy.
- Perform error analysis on misclassified positions.

---
# Team Members and Roles

**Primary Researcher / Developer**
- Dataset engineering and preprocessing  
- Neural network design and training  
- Evaluation and experimentation  
- UI implementation  
- Research documentation and paper writing  

---
# Links to documentation, code, and so on

*(To be populated during development)*

- GitHub Repository: TBD  
- Dataset Source: Lichess Puzzle Database (Kaggle)  
- Research Notes: TBD  
- Architecture Diagrams: TBD  
- Final Paper: TBD  

---
