---
marp: true
theme: default
---
# Neural Network-Based Chess Motif Detection

---
# Project Description

- Modern chess engines excel at calculating optimal moves but often fail to explain *why* those moves are strong in terms that humans understand. This project aims to bridge that gap by building a neural network capable of detecting tactical and positional motifs in chess positions.
- Using a large dataset of annotated chess puzzles, the system will analyze board states and output the motifs present in each position as a multi-label classification problem.
The end goal is to create a fast, interpretable system that enhances chess learning by identifying patterns such as forks, pins, and discovered attacks.

---
# Features and Requirements

## Core Features

### Motif Detection
**Requirements:**
- Detect well-established chess motifs (e.g., pins, forks, discovered attacks, weak squares, open files). - Completed
- Accept any valid FEN position as input. - Completed
- Achieve performance significantly better than random classification. - Completed
- Support expansion as additional motifs are introduced. - Completed

### Multi-label Output
**Requirements:**
- Allow multiple motifs per position. - Completed
- Represent outputs as a Tensor (60 motifs). - Completed
- Use sigmoid activation for independent motif probabilities. - Completed
- Evaluate using multi-label metrics such as F1 score, precision, and recall. - Completed

### Engine Analysis Integration
**Requirements:**
- Provide engine evaluations for positions. - Completed
- Display best moves alongside detected motifs. - Completed
- Use engine output as a validation mechanism for predictions. - Completed

### Readable UI
**Requirements:**
- Display chess boards in ASCII/terminal format. - COmpleted
- Output detected motifs in a clear list. - Completed
- Maintain fast response time for near real-time analysis. - Completed

---
# Data Model and Architecture

## Dataset
- Source: Lichess puzzle dataset (via Kaggle).
- File size: ~460MB CSV containing hundreds of thousands of puzzles, with millions available.
- Source: [Lichess Dataset found on Kaggle](https://www.kaggle.com/datasets/annafabris/lichess-chess-puzzles)

**Key Fields Used:**
- FEN (board state)
- Themes (motif labels)
- Moves (optional auxiliary data)

## Data Pipeline
2. Convert FEN strings into 8×8×29 tensor
3. Map puzzle themes to a standardized motif set (~60 labels).
4. Generate multi-label vectors.
5. Split dataset into training (70%), validation (15%), and test (15%).

## Neural Network Architecture

**Type:** Convolutional Neural Network (CNN)

**Approximate Size:** ~317,000 parameters  

**Training Hardware:**
- Laptop GPU (4GB GDDR6, Ampere architecture)

## Model Evaluation
- Macro F1 score
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
**Dillon Carpenter**
**Primary Developer**
- Dataset engineering and preprocessing  
- Neural network design and training  
- Evaluation and experimentation  
- UI implementation  

---
# Links to documentation, code, and so on

- GitHub Repository:https://github.com/DillonCarpenter/ASE-485-ChessMotifNN
- Dataset Source: [Lichess Dataset found on Kaggle](https://www.kaggle.com/datasets/annafabris/lichess-chess-puzzles)
---

