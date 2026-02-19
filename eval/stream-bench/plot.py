#!/usr/bin/env python3
"""
Plotting utilities for stream-bench online training results.

Generates two types of plots:
1. Test Performance: Shows accuracy progression across test windows
2. Training Progress: Shows pre/post train accuracy, improvement, and playbook growth

Can be run as a standalone script:
    python stream-bench/plot.py --run_dir results/ace_run_20260113_170913_finer_online

Or imported and used programmatically:
    from plot import plot_online_performance, plot_training_progress
    plot_online_performance(save_path, mode)
    plot_training_progress(save_path, mode)
"""
import os
import sys
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots without display
import matplotlib.pyplot as plt


def plot_offline_training_progress(save_path):
    """
    Generate training progress plots for offline mode showing pre/post train accuracy per step.

    Args:
        save_path: Path where results are saved and where plot will be saved
    """
    # Load pre_train_post_train_results
    pre_post_path = os.path.join(save_path, 'pre_train_post_train_results.json')
    if not os.path.exists(pre_post_path):
        print(f"Warning: pre_train_post_train_results.json not found at {pre_post_path}. Skipping offline plot generation.")
        return

    with open(pre_post_path, 'r') as f:
        step_results = json.load(f)

    if not step_results:
        print("Warning: Empty pre_train_post_train_results. Skipping offline plot generation.")
        return

    # Load validation results from train_results.json
    train_results_path = os.path.join(save_path, 'train_results.json')
    val_steps = []
    val_accuracies = []
    val_by_difficulty = {}  # Dictionary to track accuracy by difficulty over steps

    if os.path.exists(train_results_path):
        with open(train_results_path, 'r') as f:
            train_data = json.load(f)
            if 'results' in train_data and train_data['results']:
                for result in train_data['results']:
                    if 'val_result' in result and result['val_result']:
                        val_steps.append(result['step'])
                        val_accuracies.append(result['val_result']['accuracy'])

                        # Extract difficulty-level accuracies
                        if 'by_difficulty' in result['val_result']:
                            for difficulty, diff_data in result['val_result']['by_difficulty'].items():
                                if difficulty not in val_by_difficulty:
                                    val_by_difficulty[difficulty] = {'steps': [], 'accuracies': []}
                                val_by_difficulty[difficulty]['steps'].append(result['step'])
                                val_by_difficulty[difficulty]['accuracies'].append(diff_data['accuracy'])

    # Load final results for initial and final test accuracy
    final_results_path = os.path.join(save_path, 'final_results.json')
    initial_test_acc = None
    final_test_acc = None

    if os.path.exists(final_results_path):
        with open(final_results_path, 'r') as f:
            final_data = json.load(f)
            if 'initial_test_results' in final_data:
                initial_test_acc = final_data['initial_test_results']['accuracy']
            if 'final_test_results' in final_data:
                final_test_acc = final_data['final_test_results']['accuracy']

    # Extract data
    steps = [r['step'] for r in step_results]
    epochs = [r['epoch'] for r in step_results]
    pre_train_correct = [r['pre_train_result']['is_correct'] for r in step_results]
    post_train_correct = [r['post_train_result']['is_correct'] for r in step_results]
    playbook_tokens = [r['post_train_result']['playbook_num_tokens'] for r in step_results]
    playbook_length = [r['post_train_result']['playbook_length'] for r in step_results]
    step_times = [r.get('step_time_seconds', 0) for r in step_results]

    # Calculate cumulative accuracies
    cumulative_pre = []
    cumulative_post = []
    for i in range(len(steps)):
        cumulative_pre.append(sum(pre_train_correct[:i+1]) / (i+1))
        cumulative_post.append(sum(post_train_correct[:i+1]) / (i+1))

    # Calculate per-step improvement (1 if improved, 0 if same, -1 if worse)
    improvement = [int(post) - int(pre) for pre, post in zip(pre_train_correct, post_train_correct)]

    # Create figure with multiple subplots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Cumulative Accuracy - Pre-train vs Post-train
    ax1.plot(steps, cumulative_pre, 'r-o', linewidth=2, markersize=6, label='Pre-train (cumulative)', alpha=0.7)
    ax1.plot(steps, cumulative_post, 'g-s', linewidth=2, markersize=6, label='Post-train (cumulative)', alpha=0.7)

    # Add validation accuracy line if available
    if val_steps and val_accuracies:
        ax1.plot(val_steps, val_accuracies, 'b-^', linewidth=2, markersize=8, label='Validation Accuracy', alpha=0.8, zorder=5)

    # Add difficulty-level validation accuracy lines
    difficulty_colors = {'simple': 'lightgreen', 'moderate': 'orange', 'challenging': 'darkred'}
    difficulty_markers = {'simple': 'v', 'moderate': 'D', 'challenging': 'X'}
    for difficulty in sorted(val_by_difficulty.keys()):
        diff_data = val_by_difficulty[difficulty]
        color = difficulty_colors.get(difficulty, 'gray')
        marker = difficulty_markers.get(difficulty, 'o')
        ax1.plot(diff_data['steps'], diff_data['accuracies'],
                linestyle='--', linewidth=1.5, marker=marker, markersize=6,
                color=color, label=f'Val: {difficulty}', alpha=0.7, zorder=4)

    # Add initial and final test accuracy if available
    if initial_test_acc is not None:
        ax1.axhline(y=initial_test_acc, color='cyan', linestyle='--', linewidth=1.5, label=f'Initial Test Acc: {initial_test_acc:.3f}', alpha=0.6)
    if final_test_acc is not None:
        ax1.axhline(y=final_test_acc, color='purple', linestyle='--', linewidth=1.5, label=f'Final Test Acc: {final_test_acc:.3f}', alpha=0.6)

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Cumulative Accuracy', fontsize=12)
    ax1.set_title('Offline Mode: Training Progress (Cumulative)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    ax1.set_ylim([0, 1.0])

    # Plot 2: Per-Step Correctness (1 = improved, 0 = same, -1 = worse)
    colors = ['green' if x > 0 else 'gray' if x == 0 else 'red' for x in improvement]
    ax2.bar(steps, improvement, color=colors, alpha=0.6, edgecolor='black', width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Improvement (Post - Pre)', fontsize=12)
    ax2.set_title('Offline Mode: Per-Step Improvement', fontsize=14, fontweight='bold')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Worse', 'Same', 'Better'])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add summary statistics
    improved = sum(1 for x in improvement if x > 0)
    same = sum(1 for x in improvement if x == 0)
    worse = sum(1 for x in improvement if x < 0)
    ax2.text(0.02, 0.98, f'Improved: {improved}\nSame: {same}\nWorse: {worse}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Playbook Token Growth
    ax3.plot(steps, playbook_tokens, 'purple', marker='D', linewidth=2, markersize=6, label='Playbook Tokens')
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('Number of Tokens', fontsize=12)
    ax3.set_title('Offline Mode: Playbook Growth (Tokens)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Plot 4: Step Time Distribution
    ax4.plot(steps, step_times, 'orange', marker='o', linewidth=2, markersize=6, label='Step Time')
    ax4.axhline(y=sum(step_times)/len(step_times), color='red', linestyle='--', linewidth=2,
                label=f'Avg: {sum(step_times)/len(step_times):.1f}s')
    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.set_title('Offline Mode: Training Time per Step', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    plt.tight_layout()

    # Create plots subfolder
    plots_dir = os.path.join(save_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(plots_dir, 'offline_training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nOffline training progress plot saved to: {plot_path}")
    plt.close()

    # Save data as CSV
    csv_path = os.path.join(plots_dir, 'offline_training_data.csv')

    # Create a dictionary mapping steps to validation accuracies for easy lookup
    val_acc_by_step = {step: acc for step, acc in zip(val_steps, val_accuracies)}

    # Create dictionaries mapping steps to difficulty-level accuracies
    difficulty_acc_by_step = {}
    all_difficulties = sorted(val_by_difficulty.keys())
    for difficulty in all_difficulties:
        difficulty_acc_by_step[difficulty] = {
            step: acc for step, acc in zip(
                val_by_difficulty[difficulty]['steps'],
                val_by_difficulty[difficulty]['accuracies']
            )
        }

    # Build CSV header with difficulty columns
    header_parts = ["step", "epoch", "pre_train_correct", "post_train_correct",
                   "cumulative_pre_acc", "cumulative_post_acc", "improvement",
                   "playbook_tokens", "playbook_length", "step_time_seconds", "val_accuracy"]
    for difficulty in all_difficulties:
        header_parts.append(f"val_acc_{difficulty}")

    with open(csv_path, 'w') as f:
        f.write(",".join(header_parts) + "\n")
        for i in range(len(steps)):
            val_acc_str = f"{val_acc_by_step[steps[i]]:.4f}" if steps[i] in val_acc_by_step else ""

            # Build row
            row_parts = [
                str(steps[i]), str(epochs[i]),
                str(int(pre_train_correct[i])), str(int(post_train_correct[i])),
                f"{cumulative_pre[i]:.4f}", f"{cumulative_post[i]:.4f}",
                str(improvement[i]),
                str(playbook_tokens[i]), str(playbook_length[i]),
                f"{step_times[i]:.2f}", val_acc_str
            ]

            # Add difficulty-level accuracies
            for difficulty in all_difficulties:
                if steps[i] in difficulty_acc_by_step[difficulty]:
                    row_parts.append(f"{difficulty_acc_by_step[difficulty][steps[i]]:.4f}")
                else:
                    row_parts.append("")

            f.write(",".join(row_parts) + "\n")
    print(f"Offline training data saved to: {csv_path}")


def plot_online_performance(save_path, mode='online'):
    """
    Generate performance plots for online mode showing how accuracy changes over steps.

    Args:
        save_path: Path where results are saved and where plot will be saved
        mode: Run mode (should be 'online', default: 'online')
    """
    if mode != 'online':
        print(f"Skipping plot generation - only available for online mode (current mode: {mode})")
        return

    # Load test results from the saved JSON file (which contains window_results)
    test_results_path = os.path.join(save_path, 'test_results.json')
    if not os.path.exists(test_results_path):
        print(f"Warning: Test results file not found at {test_results_path}. Skipping plot generation.")
        return

    with open(test_results_path, 'r') as f:
        test_data = json.load(f)

    # Extract test_results from the loaded JSON
    if 'test_results' not in test_data:
        print("Warning: No test results found in test_results.json. Skipping plot generation.")
        return

    test_results = test_data['test_results']

    # Check if we have window results
    if 'window_results' not in test_results:
        print("Warning: No window results found. Skipping plot generation.")
        return

    window_results = test_results['window_results']

    # Extract window data
    window_numbers = [w['window'] for w in window_results]
    window_accuracies = [w['window_accuracy'] for w in window_results]
    window_end_indices = [w['end_idx'] for w in window_results]

    # Get initial and final test accuracy from final_results.json if available
    final_results_path = os.path.join(save_path, 'final_results.json')
    initial_test_accuracy = None
    final_test_accuracy = None

    if os.path.exists(final_results_path):
        with open(final_results_path, 'r') as f:
            final_results = json.load(f)
            if 'initial_test_results' in final_results:
                initial_test_accuracy = final_results['initial_test_results']['accuracy']
            if 'online_test_results' in final_results:
                final_test_accuracy = final_results['online_test_results']['accuracy']

    # Fallback to window data if final_results.json not available
    if initial_test_accuracy is None:
        initial_test_accuracy = window_accuracies[0] if window_accuracies else 0.0
    if final_test_accuracy is None:
        final_test_accuracy = test_results['accuracy']

    # Create figure with multiple subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Accuracy by Window
    ax1.plot(window_numbers, window_accuracies, 'b-o', linewidth=2, markersize=8, label='Window Accuracy')
    ax1.axhline(y=test_results['accuracy'], color='r', linestyle='--', linewidth=2, label=f'Overall Accuracy: {test_results["accuracy"]:.3f}')

    # Add initial test accuracy dot (before training, shown at window 0.5 to be before first window)
    ax1.plot(0.5, initial_test_accuracy, 'go', markersize=14, label=f'Initial Test Accuracy: {initial_test_accuracy:.3f}', zorder=5)
    # Add final test accuracy dot (after all training, shown at the last window)
    ax1.plot(window_numbers[-1] + 0.5, final_test_accuracy, 'ro', markersize=14, label=f'Final Test Accuracy: {final_test_accuracy:.3f}', zorder=5)

    ax1.set_xlabel('Window Number', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Online Mode: Accuracy by Training Window', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.0])

    # Add value labels on points
    for x, y in zip(window_numbers, window_accuracies):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Plot 2: Accuracy by Sample Index (cumulative)
    ax2.plot(window_end_indices, window_accuracies, 'g-s', linewidth=2, markersize=8, label='Accuracy')
    ax2.axhline(y=test_results['accuracy'], color='r', linestyle='--', linewidth=2, label=f'Overall Accuracy: {test_results["accuracy"]:.3f}')

    # Add initial test accuracy dot (at index 0, before training starts)
    ax2.plot(0, initial_test_accuracy, 'go', markersize=14, label=f'Initial Test Accuracy: {initial_test_accuracy:.3f}', zorder=5)
    # Add final test accuracy dot (at the end of all samples)
    ax2.plot(window_end_indices[-1], final_test_accuracy, 'ro', markersize=14, label=f'Final Test Accuracy: {final_test_accuracy:.3f}', zorder=5)

    ax2.set_xlabel('Sample Index (End of Window)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Online Mode: Accuracy by Sample Progress', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()

    # Create plots subfolder
    plots_dir = os.path.join(save_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(plots_dir, 'online_performance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {plot_path}")
    plt.close()

    # Also save the data as CSV for external plotting
    csv_path = os.path.join(plots_dir, 'online_performance_data.csv')
    with open(csv_path, 'w') as f:
        f.write("window,window_accuracy,start_idx,end_idx,samples_in_window\n")
        for w in window_results:
            f.write(f"{w['window']},{w['window_accuracy']},{w['start_idx']},{w['end_idx']},{w['window_total']}\n")
    print(f"Performance data saved to: {csv_path}")


def plot_training_progress(save_path, mode='online'):
    """
    Generate training progress plots showing pre/post train accuracy and playbook growth.

    Args:
        save_path: Path where results are saved and where plot will be saved
        mode: Run mode (should be 'online', default: 'online')
    """
    if mode != 'online':
        print(f"Skipping training plot generation - only available for online mode (current mode: {mode})")
        return

    # Load training results from the saved JSON file
    train_results_path = os.path.join(save_path, 'train_results.json')
    if not os.path.exists(train_results_path):
        print(f"Warning: Training results file not found at {train_results_path}. Skipping training plot generation.")
        return

    with open(train_results_path, 'r') as f:
        train_data = json.load(f)

    # Extract train_results from the loaded JSON
    if 'train_results' not in train_data:
        print("Warning: No train results found in train_results.json. Skipping training plot generation.")
        return

    train_results = train_data['train_results']

    if not train_results:
        print("Warning: Empty train results. Skipping training plot generation.")
        return

    # Load initial and final test accuracy from final_results.json if available
    final_results_path = os.path.join(save_path, 'final_results.json')
    initial_test_accuracy = None
    final_test_accuracy = None

    if os.path.exists(final_results_path):
        with open(final_results_path, 'r') as f:
            final_results = json.load(f)
            if 'initial_test_results' in final_results:
                initial_test_accuracy = final_results['initial_test_results']['accuracy']
            if 'online_test_results' in final_results:
                final_test_accuracy = final_results['online_test_results']['accuracy']

    # Fallback to test_results.json if final_results.json not available
    if initial_test_accuracy is None or final_test_accuracy is None:
        test_results_path = os.path.join(save_path, 'test_results.json')
        if os.path.exists(test_results_path):
            with open(test_results_path, 'r') as f:
                test_data = json.load(f)
                if 'test_results' in test_data and 'window_results' in test_data['test_results']:
                    window_results = test_data['test_results']['window_results']
                    if window_results:
                        if initial_test_accuracy is None:
                            initial_test_accuracy = window_results[0]['window_accuracy']
                        if final_test_accuracy is None:
                            final_test_accuracy = test_data['test_results']['accuracy']

    # Extract data from train_results
    windows = [r['window'] for r in train_results]
    pre_train_acc = [r['train_result']['pre_train_accuracy'] for r in train_results]
    post_train_acc = [r['train_result']['post_train_accuracy'] for r in train_results]
    cumulative_test_acc = [r['cumulative_test_accuracy'] for r in train_results]
    playbook_tokens = [r['playbook_num_tokens'] for r in train_results]
    playbook_length = [r['playbook_length'] for r in train_results]

    # Calculate improvement per window
    improvement = [post - pre for pre, post in zip(pre_train_acc, post_train_acc)]

    # Create figure with multiple subplots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Pre-train vs Post-train Accuracy by Window
    ax1.plot(windows, pre_train_acc, 'r-o', linewidth=2, markersize=8, label='Pre-train Accuracy')
    ax1.plot(windows, post_train_acc, 'g-s', linewidth=2, markersize=8, label='Post-train Accuracy')
    ax1.plot(windows, cumulative_test_acc, 'b--^', linewidth=2, markersize=8, label='Cumulative Test Accuracy')

    # Add initial and final test accuracy dots if available
    if initial_test_accuracy is not None and final_test_accuracy is not None:
        # Initial test accuracy shown before first window (at 0.5)
        ax1.plot(0.5, initial_test_accuracy, 'go', markersize=14, label=f'Initial Test Accuracy: {initial_test_accuracy:.3f}', zorder=5)
        # Final test accuracy shown after last window (at last window + 0.5)
        ax1.plot(windows[-1] + 0.5, final_test_accuracy, 'ro', markersize=14, label=f'Final Test Accuracy: {final_test_accuracy:.3f}', zorder=5)

    ax1.set_xlabel('Window Number', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training Progress: Pre-train vs Post-train Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.0])

    # Plot 2: Training Improvement per Window
    colors = ['green' if x >= 0 else 'red' for x in improvement]
    ax2.bar(windows, improvement, color=colors, alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Window Number', fontsize=12)
    ax2.set_ylabel('Accuracy Improvement', fontsize=12)
    ax2.set_title('Training Improvement per Window (Post - Pre)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (w, imp) in enumerate(zip(windows, improvement)):
        ax2.text(w, imp, f'{imp:+.3f}', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)

    # Plot 3: Playbook Token Growth
    ax3.plot(windows, playbook_tokens, 'purple', marker='D', linewidth=2, markersize=8, label='Playbook Tokens')
    ax3.set_xlabel('Window Number', fontsize=12)
    ax3.set_ylabel('Number of Tokens', fontsize=12)
    ax3.set_title('Playbook Growth: Token Count', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Add value labels
    for w, tokens in zip(windows, playbook_tokens):
        ax3.annotate(f'{tokens}', (w, tokens), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Plot 4: Playbook Character Length Growth
    ax4.plot(windows, playbook_length, 'orange', marker='D', linewidth=2, markersize=8, label='Playbook Length (chars)')
    ax4.set_xlabel('Window Number', fontsize=12)
    ax4.set_ylabel('Character Count', fontsize=12)
    ax4.set_title('Playbook Growth: Character Length', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Add value labels
    for w, length in zip(windows, playbook_length):
        ax4.annotate(f'{length}', (w, length), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.tight_layout()

    # Create plots subfolder
    plots_dir = os.path.join(save_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(plots_dir, 'training_progress_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining progress plot saved to: {plot_path}")
    plt.close()

    # Also save the data as CSV for external plotting
    csv_path = os.path.join(plots_dir, 'training_progress_data.csv')
    with open(csv_path, 'w') as f:
        f.write("window,pre_train_accuracy,post_train_accuracy,cumulative_test_accuracy,improvement,playbook_tokens,playbook_length\n")
        for i, w in enumerate(windows):
            f.write(f"{w},{pre_train_acc[i]},{post_train_acc[i]},{cumulative_test_acc[i]},{improvement[i]},{playbook_tokens[i]},{playbook_length[i]}\n")
    print(f"Training progress data saved to: {csv_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Generate performance plots for ACE online training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot results from a specific run directory
  python stream-bench/plot.py --run_dir results/ace_run_20260113_170913_finer_online

  # Specify mode explicitly (default is 'online')
  python stream-bench/plot.py --run_dir results/ace_run_20260113_170913_finer_online --mode online
        """
    )

    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to the ACE run directory containing test_results.json'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='online',
        choices=['online', 'offline', 'eval_only'],
        help='Run mode (default: online). Only online mode supports plotting.'
    )

    args = parser.parse_args()

    # Validate run directory exists
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory not found: {args.run_dir}")
        sys.exit(1)

    # Check if test_results.json exists (only required for online mode)
    if args.mode == 'online':
        test_results_path = os.path.join(args.run_dir, 'test_results.json')
        if not os.path.exists(test_results_path):
            print(f"Error: test_results.json not found in {args.run_dir}")
            print(f"Expected path: {test_results_path}")
            sys.exit(1)

    print(f"{'='*60}")
    print(f"GENERATING PLOTS FOR ACE RUN")
    print(f"{'='*60}")
    print(f"Run directory: {args.run_dir}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")

    # Generate plots based on mode
    if args.mode == 'online':
        print("Generating test performance plots...")
        plot_online_performance(args.run_dir, args.mode)

        print("\nGenerating training progress plots...")
        plot_training_progress(args.run_dir, args.mode)
    elif args.mode == 'offline':
        print("Generating offline training progress plots...")
        plot_offline_training_progress(args.run_dir)
    else:
        print(f"Plot generation not supported for mode: {args.mode}")

    print(f"\n{'='*60}")
    print(f"PLOTTING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
