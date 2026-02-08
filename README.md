# MixRAGRec

**This is the pytorch implementation for our MixRAGRec.**

## Architecture

MixRAGRec consists of three collaborative agents:

1. **Expert Selector Agent**: Dynamically selects the optimal retrieval strategy from four experts:
   - Expert 1: Direct Generator (no retrieval)
   - Expert 2: Triple Retriever
   - Expert 3: Subgraph Retriever
   - Expert 4: Connected Graph Retriever

2. **Knowledge Alignment Agent**: Transforms structured knowledge graph knowledge into natural language descriptions suitable for LLM processing.

3. **Recommendation Agent**: Generates personalized recommendations based on aligned knowledge.

## Training

The system is trained using **MMAPO** (Mixture-of-Experts Multi-Agent Policy Optimization), which coordinates the learning of all three agents.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Dataset

Due to space limitations, we provide a **small MovieLens-1M subset** for testing. It includes:

- **Data**: `data/movielens/ml1m_for_test.json` — 1,000 samples (train/test split by `train_ratio` in config)
- **KG**: `data/kg_test.db` — small knowledge graph (~10K entities, ~20K relations) covering movies in the test set
- **Indices**: `data/kg_indices_test/` — pre-built retrieval indices for the test KG

To run with this test set, set the dataset to `ml1m_test` in the config (see below).

### Training

```bash
python src/train.py --config configs/config.yaml
```

To use the small test dataset:

1. Open `configs/config.yaml`.
2. Set `experiment.dataset` to `ml1m_test`:
   ```yaml
   experiment:
     dataset: ml1m_test   # use small test set
     llm: llama-8b
     ...
   ```
3. Run training as above. The pipeline will load `data/movielens/ml1m_for_test.json`, `data/kg_test.db`, and `data/kg_indices_test/` automatically.

### Configuration

Edit `configs/config.yaml` to customize:
- Dataset: `ml1m_test` (small test set), `ml1m`, `lfm1k`, `ml20m`
- LLM backbone: `llama-8b`, `mistral-7b`
- Training hyperparameters
- Agent-specific settings
