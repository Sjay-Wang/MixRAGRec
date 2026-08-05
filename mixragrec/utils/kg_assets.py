"""Validation helpers for dataset-specific knowledge graph assets."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import numpy as np


REQUIRED_INDEX_FILES = (
    "entity_ids.json",
    "entity_index.npy",
    "entity_meta.json",
    "index_stats.json",
    "triple_ids.json",
    "triple_index.npy",
    "triple_meta.json",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_index_command(db_path: Path, index_dir: Path, model_name: str) -> str:
    return (
        "python -m mixragrec.kg.indexing.indexer "
        f"--db-path {db_path} --index-dir {index_dir} "
        f"--model {model_name}"
    )


def validate_kg_assets(config: Dict[str, Any]) -> Dict[str, int]:
    """Fail early when a KG database and its vector indices do not match."""
    kg_config = config.get("knowledge_graph", {})
    db_path = Path(kg_config.get("kg_db_path", ""))
    index_dir = Path(kg_config.get("kg_indices_path", ""))
    model_name = (
        config.get("models", {})
        .get("encoder", {})
        .get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    )

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Knowledge graph database not found: {db_path}. "
            "Download it and place it at the configured path."
        )

    missing_files = [
        name for name in REQUIRED_INDEX_FILES if not (index_dir / name).is_file()
    ]
    if missing_files:
        command = _build_index_command(db_path, index_dir, model_name)
        raise FileNotFoundError(
            f"Knowledge graph index is incomplete at {index_dir}. "
            f"Missing: {', '.join(missing_files)}. Build it with:\n{command}"
        )

    with sqlite3.connect(str(db_path)) as connection:
        entity_count = connection.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        triple_count = connection.execute(
            "SELECT COUNT(*) FROM relations"
        ).fetchone()[0]

    entity_ids = _load_json(index_dir / "entity_ids.json")
    entity_meta = _load_json(index_dir / "entity_meta.json")
    triple_ids = _load_json(index_dir / "triple_ids.json")
    triple_meta = _load_json(index_dir / "triple_meta.json")
    stats = _load_json(index_dir / "index_stats.json")

    entity_index = np.load(
        index_dir / "entity_index.npy", mmap_mode="r", allow_pickle=False
    )
    triple_index = np.load(
        index_dir / "triple_index.npy", mmap_mode="r", allow_pickle=False
    )

    if entity_index.ndim != 2 or triple_index.ndim != 2:
        raise ValueError("KG embedding indices must be two-dimensional arrays.")

    entity_sizes = {
        "database": entity_count,
        "entity_ids.json": len(entity_ids),
        "entity_meta.json": len(entity_meta),
        "entity_index.npy": entity_index.shape[0],
    }
    triple_sizes = {
        "database": triple_count,
        "triple_ids.json": len(triple_ids),
        "triple_meta.json": len(triple_meta),
        "triple_index.npy": triple_index.shape[0],
    }

    if len(set(entity_sizes.values())) != 1:
        raise ValueError(f"Entity index does not match the KG database: {entity_sizes}")
    if len(set(triple_sizes.values())) != 1:
        raise ValueError(f"Triple index does not match the KG database: {triple_sizes}")
    if entity_index.shape[1] != triple_index.shape[1]:
        raise ValueError(
            "Entity and triple indices use different embedding dimensions: "
            f"{entity_index.shape[1]} and {triple_index.shape[1]}"
        )

    stats_entity_count = stats.get("entity_index", {}).get("total_entities")
    stats_triple_count = stats.get("triple_index", {}).get("total_triples")
    stats_entity_dim = stats.get("entity_index", {}).get("embedding_dim")
    stats_triple_dim = stats.get("triple_index", {}).get("embedding_dim")
    if stats_entity_count != entity_count or stats_triple_count != triple_count:
        raise ValueError(
            "index_stats.json does not match the KG database: "
            f"entities={stats_entity_count}/{entity_count}, "
            f"triples={stats_triple_count}/{triple_count}"
        )
    if (
        stats_entity_dim != entity_index.shape[1]
        or stats_triple_dim != triple_index.shape[1]
    ):
        raise ValueError(
            "index_stats.json does not match the embedding dimensions: "
            f"entities={stats_entity_dim}/{entity_index.shape[1]}, "
            f"triples={stats_triple_dim}/{triple_index.shape[1]}"
        )

    stats_model = stats.get("model_name")
    if stats_model and stats_model != model_name:
        raise ValueError(
            "Index encoder does not match the configured query encoder: "
            f"{stats_model!r} != {model_name!r}"
        )

    return {
        "entities": entity_count,
        "triples": triple_count,
        "embedding_dim": entity_index.shape[1],
    }
