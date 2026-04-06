# Base image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install Python packages
RUN pip install --no-cache-dir pandas numpy python-chess scikit-learn

COPY chess_engines/stockfish-ubuntu-x86-64-avx2 /usr/local/bin/stockfish
RUN chmod +x /usr/local/bin/stockfish
# Set default workdir (Already set in base image)
# WORKDIR /workspace