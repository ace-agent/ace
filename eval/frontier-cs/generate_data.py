#!/usr/bin/env python3
"""
Generate Frontier-CS JSONL datasets in ACE format.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_algorithmic_samples(frontier_root: Path) -> List[Dict]:
    problems_dir = frontier_root / "algorithmic" / "problems"
    samples = []

    for problem_dir in sorted(problems_dir.iterdir()):
        if not problem_dir.is_dir() or not problem_dir.name.isdigit():
            continue

        statement_path = problem_dir / "statement.txt"
        if not statement_path.exists():
            continue

        statement = read_text(statement_path).strip()
        config_path = problem_dir / "config.yaml"

        context = statement

        target = ""

        samples.append({
            "context": context,
            "target": target,
            "metadata": {
                "track": "algorithmic",
                "problem_id": int(problem_dir.name),
                "statement_path": str(statement_path),
                "config_path": str(config_path) if config_path.exists() else "",
            },
        })

    return samples


# def build_research_samples(frontier_root: Path) -> List[Dict]:
#     problems_dir = frontier_root / "research" / "problems"
#     solutions_dir = frontier_root / "research" / "solutions"
#     samples = []

#     for readme_path in sorted(problems_dir.rglob("readme")):
#         if not readme_path.is_file():
#             continue

#         problem_dir = readme_path.parent
#         problem_id = problem_dir.relative_to(problems_dir).as_posix()

#         readme = read_text(readme_path).strip()

#         config_path = problem_dir / "config.yaml"
#         config_text = read_text(config_path).strip() if config_path.exists() else ""

#         context = readme
#         if config_text:
#             context = f"{context}\n\n[config]\n{config_text}"

#         target = ""
#         solution_dir = solutions_dir / problem_id
#         has_reference = False
#         if solution_dir.exists():
#             has_reference = any(solution_dir.glob("*.py"))

#         reference_note = "not_found"
#         if has_reference:
#             reference_note = "omitted_by_design"

#         samples.append({
#             "context": context,
#             "target": target,
#             "metadata": {
#                 "track": "research",
#                 "problem_id": problem_id,
#                 "readme_path": str(readme_path),
#                 "config_path": str(config_path) if config_path.exists() else "",
#                 "reference_included": reference_note == "included",
#                 "reference_note": reference_note,
#             },
#         })

#     return samples


def split_samples(samples: List[Dict], seed: int, train_ratio: float,
                  val_ratio: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)
    rng.shuffle(samples)

    total = len(samples)
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))
    test_count = total - train_count - val_count

    if test_count <= 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        else:
            val_count = max(1, val_count - 1)

    train = samples[:train_count]
    val = samples[train_count:train_count + val_count]
    test = samples[train_count + val_count:]
    return train, val, test


def write_jsonl(items: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Frontier-CS datasets for ACE")
    parser.add_argument(
        "--frontier_root",
        type=str,
        default="/Users/jingzhuohu/repos/Frontier-CS",
        help="Path to Frontier-CS repo",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/jingzhuohu/repos/ace/eval/frontier-cs/data",
        help="Output directory for JSONL files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    frontier_root = Path(args.frontier_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algorithmic_samples = build_algorithmic_samples(frontier_root)
    # research_samples = build_research_samples(frontier_root)

    algo_train, algo_val, algo_test = split_samples(
        algorithmic_samples, args.seed, args.train_ratio, args.val_ratio
    )
    # res_train, res_val, res_test = split_samples(
    #     research_samples, args.seed, args.train_ratio, args.val_ratio
    # )

    algo_train_path = output_dir / "algorithmic_train.jsonl"
    algo_val_path = output_dir / "algorithmic_val.jsonl"
    algo_test_path = output_dir / "algorithmic_test.jsonl"
    res_train_path = output_dir / "research_train.jsonl"
    res_val_path = output_dir / "research_val.jsonl"
    res_test_path = output_dir / "research_test.jsonl"

    write_jsonl(algo_train, algo_train_path)
    write_jsonl(algo_val, algo_val_path)
    write_jsonl(algo_test, algo_test_path)
    # write_jsonl(res_train, res_train_path)
    # write_jsonl(res_val, res_val_path)
    # write_jsonl(res_test, res_test_path)

    sample_config = {
        "algorithmic": {
            "train_data": "./eval/frontier-cs/data/algorithmic_train.jsonl",
            "val_data": "./eval/frontier-cs/data/algorithmic_val.jsonl",
            "test_data": "./eval/frontier-cs/data/algorithmic_test.jsonl",
        },
        "research": {
            "train_data": "./eval/frontier-cs/data/research_train.jsonl",
            "val_data": "./eval/frontier-cs/data/research_val.jsonl",
            "test_data": "./eval/frontier-cs/data/research_test.jsonl",
        },
    }

    config_path = output_dir / "sample_config.json"
    config_path.write_text(json.dumps(sample_config, indent=2), encoding="utf-8")

    print("Generated Frontier-CS datasets:")
    print(f"  algorithmic: {len(algorithmic_samples)} samples")
    # print(f"  research:    {len(research_samples)} samples")
    print(f"  output_dir:  {output_dir}")


if __name__ == "__main__":
    main()

