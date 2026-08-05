import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixragrec.utils.kg_assets import validate_kg_assets


class KGAssetValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_path = root / "kg.db"
        self.index_dir = root / "indices"
        self.index_dir.mkdir()

        with sqlite3.connect(self.db_path) as connection:
            connection.execute("CREATE TABLE entities (entity_id TEXT PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE relations (relation_id TEXT PRIMARY KEY)"
            )
            connection.executemany(
                "INSERT INTO entities VALUES (?)", [("e1",), ("e2",)]
            )
            connection.execute("INSERT INTO relations VALUES ('r1')")

        self._write_json("entity_ids.json", ["e1", "e2"])
        self._write_json("entity_meta.json", [{}, {}])
        self._write_json("triple_ids.json", ["r1"])
        self._write_json("triple_meta.json", [{}])
        np.save(self.index_dir / "entity_index.npy", np.zeros((2, 3), dtype=np.float32))
        np.save(self.index_dir / "triple_index.npy", np.zeros((1, 3), dtype=np.float32))
        self._write_json(
            "index_stats.json",
            {
                "entity_index": {"total_entities": 2, "embedding_dim": 3},
                "triple_index": {"total_triples": 1, "embedding_dim": 3},
                "model_name": "test-model",
            },
        )

        self.config = {
            "knowledge_graph": {
                "kg_db_path": str(self.db_path),
                "kg_indices_path": str(self.index_dir),
            },
            "models": {"encoder": {"model_name": "test-model"}},
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, name, value):
        with (self.index_dir / name).open("w", encoding="utf-8") as file:
            json.dump(value, file)

    def test_matching_assets_are_accepted(self):
        summary = validate_kg_assets(self.config)
        self.assertEqual(
            summary, {"entities": 2, "triples": 1, "embedding_dim": 3}
        )

    def test_database_index_mismatch_is_rejected(self):
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("INSERT INTO entities VALUES ('e3')")

        with self.assertRaisesRegex(ValueError, "Entity index does not match"):
            validate_kg_assets(self.config)


if __name__ == "__main__":
    unittest.main()
