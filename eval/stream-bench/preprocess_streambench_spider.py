#!/usr/bin/env python3
"""
Download StreamBench (Spider subset) from Hugging Face and preprocess to:
(1) supplement database schema (db_schema) using Spider tables.json (preferred)
    with a fallback to SQLite introspection if tables.json is missing/doesn't match.
(2) output only: question_id, question, sql, difficulty, db_name, db_schema

Examples:
  # Generate test split using Spider databases
  python preprocess_streambench_spider.py \
    --spider_root ./data/spider \
    --tables_json ./data/spider/tables.json \
    --split test \
    --out ./data/streambench_spider_test.jsonl

  # Generate train split using Spider databases
  python preprocess_streambench_spider.py \
    --spider_root ./data/spider \
    --tables_json ./data/spider/tables.json \
    --split train \
    --out ./data/streambench_spider_train.jsonl

  # Generate validation split
  python preprocess_streambench_spider.py \
    --spider_root ./data/spider \
    --tables_json ./data/spider/tables.json \
    --split validation \
    --out ./data/streambench_spider_val.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

def load_streambench_spider(split: str):
    # Hugging Face datasets
    from datasets import load_dataset
    # Subset name is "spider", splits include train/validation/test
    ds = load_dataset("appier-ai-research/StreamBench", "spider", split=split)
    return ds

def read_tables_json(tables_json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Read Spider-style tables.json:
    Each entry has db_id, table_names_original, column_names_original, column_types, foreign_keys, primary_keys, etc.
    Format described in Spider docs and used widely by text-to-SQL datasets.
    """
    with open(tables_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dbid_to_schema: Dict[str, Dict[str, Any]] = {}
    for db in data:
        db_id = db.get("db_id")
        if not db_id:
            continue

        table_names = db.get("table_names_original", [])
        column_names = db.get("column_names_original", [])
        column_types = db.get("column_types", [])

        # column_names: list like [[table_idx, "colname"], ...]
        # table_idx = -1 for "*"
        tables: List[Dict[str, Any]] = [{"name": t, "columns": []} for t in table_names]

        for i, (tbl_idx, col_name) in enumerate(column_names):
            if tbl_idx == -1:
                continue
            col_type = column_types[i] if i < len(column_types) else None
            tables[tbl_idx]["columns"].append({"name": col_name, "type": col_type})

        schema_obj = {
            "db_id": db_id,
            "tables": tables,
            "primary_keys": db.get("primary_keys", []),
            "foreign_keys": db.get("foreign_keys", []),
        }
        dbid_to_schema[db_id] = schema_obj

    return dbid_to_schema

def find_sqlite_path(spider_root: str, db_id: str) -> Optional[str]:
    """
    Try common Spider layouts. We search for a .sqlite file under spider_root containing db_id.
    Typical Spider structure: spider/database/{db_id}/{db_id}.sqlite
    """
    patterns = [
        os.path.join(spider_root, "database", db_id, "*.sqlite"),
        os.path.join(spider_root, "database", f"{db_id}.sqlite"),
        os.path.join(spider_root, "database", db_id, f"{db_id}.sqlite"),
        os.path.join(spider_root, "**", db_id, "*.sqlite"),
        os.path.join(spider_root, "**", f"{db_id}.sqlite"),
        os.path.join(spider_root, "**", db_id, f"{db_id}.sqlite"),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            # prefer shortest path / deterministic choice
            hits.sort(key=lambda p: (len(p), p))
            return hits[0]
    return None

def sqlite_introspect_schema(sqlite_path: str) -> Dict[str, Any]:
    """
    Build a minimal schema object from SQLite:
      - table names
      - columns + types
      - foreign keys
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    table_names = [r["name"] for r in cur.fetchall()]

    tables: List[Dict[str, Any]] = []
    foreign_keys: List[List[int]] = []  # keep same shape as tables.json-ish: [ [src_col_global, dst_col_global], ... ] (best-effort)
    primary_keys: List[int] = []         # best-effort global column indices

    # We'll store columns globally as we go, to emulate tables.json indices roughly.
    # This is a fallback only; for training it's usually fine to use structured per-table schema.
    global_col_index: Dict[Tuple[str, str], int] = {}
    global_cols: List[Tuple[str, str]] = []

    for t in table_names:
        cur.execute(f"PRAGMA table_info('{t}');")
        cols = []
        for row in cur.fetchall():
            col_name = row["name"]
            col_type = row["type"]
            is_pk = int(row["pk"]) == 1
            cols.append({"name": col_name, "type": col_type})

            idx = len(global_cols)
            global_cols.append((t, col_name))
            global_col_index[(t, col_name)] = idx
            if is_pk:
                primary_keys.append(idx)

        tables.append({"name": t, "columns": cols})

    # Foreign keys (best-effort mapping to global column indices)
    for t in table_names:
        cur.execute(f"PRAGMA foreign_key_list('{t}');")
        for fk in cur.fetchall():
            src_col = fk["from"]
            dst_table = fk["table"]
            dst_col = fk["to"]
            if (t, src_col) in global_col_index and (dst_table, dst_col) in global_col_index:
                foreign_keys.append([global_col_index[(t, src_col)], global_col_index[(dst_table, dst_col)]])

    conn.close()
    return {
        "sqlite_path": sqlite_path,
        "tables": tables,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }

def coerce_row(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    StreamBench Spider fields: db_id, question, SQL, question_id, difficulty (if available), evidence.
    Map to standardized format compatible with the data processor.
    """
    db_id = example.get("db_id")
    qid = example.get("question_id")
    question = example.get("question")
    sql = example.get("SQL")  # note: uppercase in StreamBench
    difficulty = example.get("difficulty", "unknown")  # Spider might not have difficulty in all versions
    return {
        "question_id": qid,
        "question": question,
        "sql": sql,
        "difficulty": difficulty,
        "db_name": db_id,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--out", default="streambench_spider_test.jsonl")
    ap.add_argument("--tables_json", default="", help="Path to Spider-style tables.json (recommended).")
    ap.add_argument("--spider_root", default="", help="Root dir where Spider sqlite databases live (fallback if tables.json missing).")
    ap.add_argument("--schema_format", default="json", choices=["json", "string"],
                    help="Store db_schema as structured JSON (json) or as a compact string (string).")
    args = ap.parse_args()

    # Load dataset
    ds = load_streambench_spider(args.split)

    # Load schema map from tables.json if provided
    dbid_to_schema = {}
    if args.tables_json:
        if not os.path.exists(args.tables_json):
            raise FileNotFoundError(f"--tables_json not found: {args.tables_json}")
        dbid_to_schema = read_tables_json(args.tables_json)

    # Process and write
    n_missing_schema = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            row = coerce_row(ex)
            db_id = row["db_name"]

            schema_obj = dbid_to_schema.get(db_id)

            # Fallback to sqlite introspection if needed
            if schema_obj is None:
                if args.spider_root:
                    sqlite_path = find_sqlite_path(args.spider_root, db_id)
                    if sqlite_path:
                        schema_obj = sqlite_introspect_schema(sqlite_path)

            if schema_obj is None:
                n_missing_schema += 1
                schema_obj = {"error": "schema_not_found", "db_id": db_id}

            if args.schema_format == "string":
                # Compact, readable schema string
                if "tables" in schema_obj and isinstance(schema_obj["tables"], list):
                    parts = []
                    for t in schema_obj["tables"]:
                        tname = t.get("name", "")
                        cols = t.get("columns", [])
                        col_str = ", ".join([c.get("name", "") for c in cols if isinstance(c, dict)])
                        parts.append(f"{tname}({col_str})")
                    row["db_schema"] = "\n".join(parts)
                else:
                    row["db_schema"] = json.dumps(schema_obj, ensure_ascii=False)
            else:
                row["db_schema"] = schema_obj

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.out}")
    if n_missing_schema:
        print(f"WARNING: {n_missing_schema} rows had missing schema (schema_not_found). Provide --tables_json and/or --spider_root.")

if __name__ == "__main__":
    main()
