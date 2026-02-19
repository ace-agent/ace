#!/usr/bin/env python3
"""
Dataset Statistics Script

This script analyzes and displays statistics for BIRD, CoSQL, and Spider datasets.
It shows total samples and difficulty distribution for train, val, and test splits.

Usage:
    python dataset_stats.py <dataset_name>

where <dataset_name> is one of: bird, cosql, spider
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_difficulty_stats(data):
    """Calculate difficulty distribution from dataset."""
    difficulty_counts = defaultdict(int)
    for item in data:
        difficulty = item.get('difficulty', 'unknown')
        difficulty_counts[difficulty] += 1
    return dict(difficulty_counts)


def print_split_stats(split_name, data):
    """Print statistics for a single data split."""
    total = len(data)
    difficulty_stats = get_difficulty_stats(data)

    print(f"\n{split_name.upper()}:")
    print(f"  Total samples: {total}")

    if difficulty_stats:
        print(f"  By difficulty:")
        # Sort difficulties for consistent output
        for difficulty in sorted(difficulty_stats.keys()):
            count = difficulty_stats[difficulty]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {difficulty}: {count} ({percentage:.1f}%)")


def get_dataset_paths(dataset_name):
    """Get file paths for the specified dataset."""
    base_path = Path(__file__).parent / "data"

    paths = {
        'train': base_path / f"streambench_{dataset_name}_train.jsonl",
        'val': base_path / f"streambench_{dataset_name}_val.jsonl",
        'test': base_path / f"streambench_{dataset_name}_test.jsonl"
    }

    return paths


def print_dataset_stats(dataset_name):
    """Print comprehensive statistics for a dataset."""
    print(f"=" * 60)
    print(f"Dataset Statistics: {dataset_name.upper()}")
    print(f"=" * 60)

    paths = get_dataset_paths(dataset_name)

    # Track totals across all splits
    total_samples = 0
    total_difficulty = defaultdict(int)

    for split_name in ['train', 'val', 'test']:
        file_path = paths[split_name]

        if not file_path.exists():
            print(f"\n{split_name.upper()}: File not found at {file_path}")
            continue

        try:
            data = load_jsonl(file_path)
            print_split_stats(split_name, data)

            # Update totals
            total_samples += len(data)
            difficulty_stats = get_difficulty_stats(data)
            for difficulty, count in difficulty_stats.items():
                total_difficulty[difficulty] += count

        except Exception as e:
            print(f"\n{split_name.upper()}: Error loading file - {e}")

    # Print overall statistics
    print(f"\n{'-' * 60}")
    print(f"OVERALL STATISTICS:")
    print(f"  Total samples across all splits: {total_samples}")

    if total_difficulty:
        print(f"  Overall difficulty distribution:")
        for difficulty in sorted(total_difficulty.keys()):
            count = total_difficulty[difficulty]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"    {difficulty}: {count} ({percentage:.1f}%)")

    print(f"=" * 60)


def main():
    """Main function to parse arguments and display statistics."""
    if len(sys.argv) != 2:
        print("Error: Dataset name required")
        print("\nUsage: python dataset_stats.py <dataset_name>")
        print("where <dataset_name> is one of: bird, cosql, spider")
        sys.exit(1)

    dataset_name = sys.argv[1].lower()

    valid_datasets = ['bird', 'cosql', 'spider']
    if dataset_name not in valid_datasets:
        print(f"Error: Invalid dataset name '{dataset_name}'")
        print(f"Valid options: {', '.join(valid_datasets)}")
        sys.exit(1)

    print_dataset_stats(dataset_name)


if __name__ == "__main__":
    main()
