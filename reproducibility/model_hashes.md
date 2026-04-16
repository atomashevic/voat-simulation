# Model Provenance and Version Information

This document provides version information and download sources for all models used in the simulation and analysis pipelines.

## Simulation Models

### Dolphin Mistral 24B Venice Edition

- **Purpose**: Base LLM for agent text generation
- **Source**: Ollama registry
- **Pull command**: `ollama pull ikiru/Dolphin-Mistral-24B-Venice-Edition`
- **HuggingFace source**: [cognitivecomputations/Dolphin-Mistral-24B-Venice-Edition](https://huggingface.co/cognitivecomputations/Dolphin-Mistral-24B-Venice-Edition)
- **Ollama page**: https://ollama.com/ikiru/Dolphin-Mistral-24B-Venice-Edition
- **Model family**: Mistral 24B (Dolphin fine-tune)
- **Quantization**: Q4_K_M (Ollama default)

**Verification**: After pulling the model, verify with:
```bash
ollama show ikiru/Dolphin-Mistral-24B-Venice-Edition --modelfile
```

## Analysis Models

### all-MiniLM-L6-v2

- **Purpose**: Sentence embeddings for topic modeling and semantic similarity
- **Source**: HuggingFace sentence-transformers
- **Model ID**: `sentence-transformers/all-MiniLM-L6-v2`
- **URL**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Recommended commit**: Use latest from `main` branch or pin to specific commit

**Python usage**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### bert-base-uncased

- **Purpose**: Tokenization and embeddings for convergence entropy analysis
- **Source**: HuggingFace transformers
- **Model ID**: `bert-base-uncased`
- **URL**: https://huggingface.co/bert-base-uncased
- **Recommended commit**: Use latest from `main` branch or pin to specific commit

**Python usage**:
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
```

### ToxiGen RoBERTa

- **Purpose**: Toxicity classification
- **Source**: HuggingFace
- **Model ID**: `tomh/toxigen_roberta`
- **URL**: https://huggingface.co/tomh/toxigen_roberta
- **Training data**: ToxiGen benchmark dataset

**Python usage**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('tomh/toxigen_roberta')
model = AutoModelForSequenceClassification.from_pretrained('tomh/toxigen_roberta')
```

## Version Pinning

For strict reproducibility, we recommend pinning HuggingFace models to specific commits:

```python
from sentence_transformers import SentenceTransformer

# Pin to specific commit (example)
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    revision='main'  # Replace with specific commit hash for strict reproducibility
)
```

## Model Download Script

To download all required models before running analyses:

```bash
# Download sentence-transformers model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Download BERT model
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"

# Download ToxiGen model
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('tomh/toxigen_roberta'); AutoModelForSequenceClassification.from_pretrained('tomh/toxigen_roberta')"

# Download Ollama model (requires Ollama installed)
ollama pull ikiru/Dolphin-Mistral-24B-Venice-Edition
```

## Hardware Requirements

- **Dolphin Mistral 24B**: Requires ~24GB VRAM for inference (tested on NVIDIA A30)
- **Analysis models**: Can run on CPU; GPU recommended for batch processing
- **Total GPU memory for analysis**: ~4-8GB recommended

