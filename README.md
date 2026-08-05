# MixRAGRec

PyTorch implementation of our KDD paper, **"Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation."**

## Overview

MixRAGRec coordinates three agents:

1. **Expert Selector Agent** selects one of four knowledge retrieval strategies:
   direct generation, triple retrieval, subgraph retrieval, or connected-graph
   retrieval.
2. **Knowledge Alignment Agent** converts the retrieved graph knowledge into
   text suitable for an LLM.
3. **Recommendation Agent** predicts the preferred item from the candidate set.

The agents are optimized with MMAPO (Mixture-of-Experts Multi-Agent Policy Optimization).

## Installation

```bash
pip install -r requirements.txt
```

## Data And Knowledge Graphs

The repository contains the recommendation data and item-to-DBpedia mappings for all supported datasets. The small test KG and the MovieLens-1M KG indices are also included. The larger dataset-specific databases are available from the following Google Drive folder:

<https://drive.google.com/drive/folders/1-CpalZvGRzqjBIwwDtOoO4d3uU9OlR4C?usp=sharing>

| Dataset | Recommendation data | KG database | Vector indices |
| --- | --- | --- | --- |
| `ml1m_test` | `data/movielens/ml1m_for_test.json` | `data/kg_test.db` | `data/kg_indices_test/` |
| `ml1m` | `data/movielens/ml1m.json` | `data/parsed_kg_from_dump.db` | `data/kg_indices/` |
| `lfm1k` | `data/LFM/lfm1k.json` | Download `parsed_kg_lastfm.db` | Generate `data/kg_indices_lfm1k/` |
| `ml20m` | `data/movielens20M/ml20m.json` | Download `parsed_kg_ml20m.db` | Generate `data/kg_indices_ml20m/` |

Place the downloaded databases at these exact paths:

```text
data/parsed_kg_lastfm.db
data/parsed_kg_ml20m.db
```

## Build Indices For LFM1K And ML20M

Run the commands from the repository root. The index encoder must match the query encoder configured in `configs/config.yaml`.

### LFM1K

```bash
python -m mixragrec.kg.indexing.indexer \
  --db-path data/parsed_kg_lastfm.db \
  --index-dir data/kg_indices_lfm1k \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 \
  --device cuda
```

### MovieLens-20M

```bash
python -m mixragrec.kg.indexing.indexer \
  --db-path data/parsed_kg_ml20m.db \
  --index-dir data/kg_indices_ml20m \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128 \
  --device cuda
```

Each command creates:

```text
entity_ids.json
entity_index.npy
entity_meta.json
index_stats.json
triple_ids.json
triple_index.npy
triple_meta.json
```

Successful indexing should report the following counts:

| Dataset | Entities | Triples | Embedding dimension |
| --- | ---: | ---: | ---: |
| LFM1K | 180,457 | 516,604 | 384 |
| MovieLens-20M | 72,673 | 272,173 | 384 |

## Quick Start

For a lightweight pipeline check, select `ml1m_test` in `configs/config.yaml`:

```yaml
experiment:
  dataset: ml1m_test
  llm: llama-8b
```

Then run:

```bash
python src/train.py --config configs/config.yaml
```

The test configuration uses 1,000 recommendation samples, `data/kg_test.db`, and the prebuilt `data/kg_indices_test/` directory.

## Full Experiments

Set `experiment.dataset` to one of `ml1m`, `lfm1k`, or `ml20m`:

```yaml
experiment:
  dataset: lfm1k
  llm: llama-8b
```

Start training with:

```bash
python src/train.py --config configs/config.yaml
```

To evaluate a saved checkpoint:

```bash
python src/test.py --best
```

### 🌹Please Cite Our Work If Helpful:



**Thanks! / 谢谢! / ありがとう! / merci! / 감사! / Danke! / спасибо! / gracias! ...**



```bibtex
@article{wang2026mixture,
  title={Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation},
  author={Wang, Shijie and Liu, Chengyi and Ding, Yujuan and Lin, Shanru and Ng, See-Kiong and Xin, Xu and Fan, Wenqi},
  journal={arXiv preprint arXiv:2605.28175},
  year={2026}
}
```
