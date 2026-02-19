#!/usr/bin/env python3
"""
Script to analyze errors from ACE terminal log files.
Counts errors during initial and final test accuracy calculations and classifies error types.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


class ErrorAnalyzer:
    def __init__(self, log_file_path: str):
        self.log_file_path = Path(log_file_path)
        self.initial_errors = defaultdict(list)
        self.final_errors = defaultdict(list)
        self.between_errors = defaultdict(list)  # Errors between initial and final
        self.current_phase = None  # 'initial', 'final', or 'between'
        self.current_window = None  # Track current window number
        self.window_accuracies = {}  # Track window accuracies {window_num: accuracy}
        self.cumulative_accuracies = {}  # Track cumulative test accuracies {window_num: cumulative_acc}
        self.cumulative_samples = {}  # Track cumulative sample counts {window_num: total_samples}
        self.window_sizes = {}  # Track window sizes {window_num: num_samples}
        self.default_window_size = None  # Default window size if not specified per window

        # Track component-specific errors (generator, reflector, curator)
        self.component_errors = defaultdict(lambda: defaultdict(list))  # {phase: {component: [errors]}}

        # Track which problems/samples have errors
        self.problem_errors = defaultdict(set)  # {phase: set of (window, sample_idx/call_id)}

        # Track current sample being processed
        self.current_sample_idx = None
        self.current_call_id = None

    def classify_error(self, error_line: str, details: str = "") -> str:
        """Classify the type of error based on the error message."""
        combined_text = error_line + " " + details

        # Context length errors
        if "context_length_exceeded" in combined_text or "tokens exceed" in combined_text:
            return "context_length_exceeded"

        # Rate limit errors
        if "rate_limit" in combined_text.lower() or "429" in combined_text:
            return "rate_limit"

        # Invalid request errors (excluding context length)
        if "invalid_request_error" in combined_text or "invalid prompt" in combined_text:
            if "context_length" not in combined_text:
                return "invalid_request"

        # Authentication errors
        if "authentication" in combined_text.lower() or "401" in combined_text:
            return "authentication_error"

        # Server errors
        if "500" in combined_text or "502" in combined_text or "503" in combined_text:
            return "server_error"

        # Timeout errors
        if "timeout" in combined_text.lower():
            return "timeout_error"

        # Connection errors
        if "connection" in combined_text.lower():
            return "connection_error"

        # Client errors (general)
        if "client error" in combined_text.lower():
            return "client_error_other"

        # Unknown errors
        return "unknown_error"

    def extract_component(self, line: str) -> str:
        """Extract the component name (GENERATOR, REFLECTOR, CURATOR) from the log line."""
        if '[GENERATOR]' in line:
            return 'GENERATOR'
        elif '[REFLECTOR]' in line:
            return 'REFLECTOR'
        elif '[CURATOR]' in line:
            return 'CURATOR'
        return 'UNKNOWN'

    def extract_base_sample_id(self, call_id: str) -> str:
        """Extract base sample ID from call ID.

        Examples:
            'online_train_s_1455_round_2' -> '1455'
            'online_train_s_1320_post_curate' -> '1320'
            'gen_call_1' -> 'gen_call_1'
            '123' -> '123'
        """
        # Try to match patterns like "s_1455_round_2" or "s_1320_post_curate"
        match = re.search(r's_(\d+)(?:_round_\d+|_post_curate|_pre_curate)?', call_id)
        if match:
            return match.group(1)

        # If no pattern match, return the original (might be sample_idx)
        return call_id

    def detect_phase(self, line: str) -> None:
        """Detect whether we're in the initial or final test accuracy calculation phase."""
        # Track window size (appears early in logs as "Window size: N")
        window_size_match = re.search(r'Window size:\s*(\d+)', line, re.IGNORECASE)
        if window_size_match:
            self.default_window_size = int(window_size_match.group(1))

        # Track window numbers - these set the context for what window we're in
        window_match = re.search(r'WINDOW (\d+)', line, re.IGNORECASE)
        if window_match:
            self.current_window = int(window_match.group(1))

        # Also check for "Testing window X" pattern
        testing_window_match = re.search(r'Testing window (\d+)', line, re.IGNORECASE)
        if testing_window_match:
            self.current_window = int(testing_window_match.group(1))

        # Track window step counts like "Window 1, Step 5/15"
        window_step_match = re.search(r'Window\s+(\d+),\s+Step\s+\d+/(\d+)', line, re.IGNORECASE)
        if window_step_match:
            window_num = int(window_step_match.group(1))
            window_size = int(window_step_match.group(2))
            self.window_sizes[window_num] = window_size

        # Track sample indices and call IDs
        sample_match = re.search(r'sample[_ ]?(?:idx|index|#)?[:\s]+(\d+)', line, re.IGNORECASE)
        if sample_match:
            self.current_sample_idx = int(sample_match.group(1))

        # Track call IDs from component logs like "[GENERATOR] Call XXX"
        call_match = re.search(r'\[(GENERATOR|REFLECTOR|CURATOR)\]\s+Call\s+([A-Za-z0-9_-]+)', line)
        if call_match:
            self.current_call_id = call_match.group(2)

        # Track window accuracy: "Window X test accuracy: Y.YYY"
        # This pattern appears after testing each window and tells us which window completed
        accuracy_match = re.search(r'Window\s+(\d+)\s+test\s+accuracy:\s+([\d.]+)', line, re.IGNORECASE)
        if accuracy_match:
            window_num = int(accuracy_match.group(1))
            accuracy = float(accuracy_match.group(2))
            self.window_accuracies[window_num] = accuracy
            # Update current window to match the window that just reported accuracy
            self.current_window = window_num

        # Track cumulative test accuracy: "Cumulative test accuracy so far: Y.YYY (N samples)"
        # This appears right after the window accuracy line
        # IMPORTANT: This cumulative accuracy should be associated with the window that just completed
        cumulative_match = re.search(r'Cumulative\s+test\s+accuracy\s+so\s+far:\s+([\d.]+)\s+\((\d+)\s+samples\)', line, re.IGNORECASE)
        if cumulative_match:
            cumulative_acc = float(cumulative_match.group(1))
            total_samples = int(cumulative_match.group(2))
            # Associate with current window (which should have been set by the previous "Window X test accuracy" line)
            if self.current_window:
                self.cumulative_accuracies[self.current_window] = cumulative_acc
                self.cumulative_samples[self.current_window] = total_samples

        # Look for phase indicators in the log
        if "initial test acc" in line.lower() or "calculating initial" in line.lower():
            self.current_phase = "initial"
        elif "final test acc" in line.lower() or "calculating final" in line.lower():
            self.current_phase = "final"
        elif "test acc" in line.lower() and self.current_phase is None:
            # If we haven't seen initial yet, assume we're in initial phase
            self.current_phase = "initial"
        elif self.current_phase == "initial" and "done" in line.lower():
            # Initial phase completed, now in between phase
            self.current_phase = "between"

    def analyze_log(self) -> None:
        """Parse the log file and categorize errors."""
        if not self.log_file_path.exists():
            print(f"Error: Log file not found at {self.log_file_path}")
            sys.exit(1)

        with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Detect phase changes
            self.detect_phase(line)

            # Look for error indicators
            # Match both ⚠️ errors and "failed after" errors
            is_warning_error = "⚠️" in line and ("error" in line.lower() or "Error" in line)
            is_failed_error = "failed after" in line and ("Error" in line or "error" in line)

            if is_warning_error or is_failed_error:
                # Extract error details from the current line
                error_msg = line

                # Extract component that caused the error
                component = self.extract_component(error_msg)

                # Look ahead for [GENERATOR] Error details on the next line(s)
                details = ""
                j = i + 1
                while j < len(lines) and j < i + 5:  # Look ahead up to 5 lines
                    next_line = lines[j].strip()
                    if "[GENERATOR] Error details:" in next_line or "[REFLECTOR] Error details:" in next_line or "[CURATOR] Error details:" in next_line or "Error code:" in next_line:
                        details += " " + next_line
                        j += 1
                        # Also extract component from error details if not found yet
                        if component == 'UNKNOWN':
                            component = self.extract_component(next_line)
                    elif next_line and not next_line.startswith("["):
                        # Continuation of error details
                        details += " " + next_line
                        j += 1
                    else:
                        break

                # Classify the error
                error_type = self.classify_error(error_msg, details)

                # Determine phase for tracking
                phase = self.current_phase if self.current_phase else "initial"

                # Store in appropriate phase
                error_entry = {
                    'line_num': i + 1,
                    'message': error_msg,
                    'details': details.strip(),
                    'window': self.current_window,
                    'component': component,
                    'sample_idx': self.current_sample_idx,
                    'call_id': self.current_call_id
                }

                # Track which problems have errors
                problem_key = (self.current_window, self.current_sample_idx or self.current_call_id)
                if problem_key[1] is not None:  # Only track if we have a sample/call identifier
                    self.problem_errors[phase].add(problem_key)

                # Store in component-specific tracking
                self.component_errors[phase][component].append(error_entry)

                if self.current_phase == "final":
                    self.final_errors[error_type].append(error_entry)
                elif self.current_phase == "between":
                    self.between_errors[error_type].append(error_entry)
                else:
                    # Default to initial if phase is unknown
                    self.initial_errors[error_type].append(error_entry)

            i += 1

    def print_component_analysis(self, phase: str, phase_name: str) -> None:
        """Print component-specific error analysis for a given phase."""
        component_stats = self.component_errors.get(phase, {})
        if not component_stats:
            return

        print(f"\nComponent breakdown:")
        total_component_errors = sum(len(errors) for errors in component_stats.values())

        for component in ['GENERATOR', 'REFLECTOR', 'CURATOR', 'UNKNOWN']:
            errors = component_stats.get(component, [])
            if errors:
                percentage = (len(errors) / total_component_errors * 100) if total_component_errors > 0 else 0
                print(f"  • {component}: {len(errors)} errors ({percentage:.1f}%)")

                # Count unique problems for this component
                unique_problems = set()
                for error in errors:
                    problem_key = (error['window'], error['sample_idx'] or error['call_id'])
                    if problem_key[1] is not None:
                        unique_problems.add(problem_key)

                if unique_problems:
                    avg_errors_per_problem = len(errors) / len(unique_problems)
                    print(f"    - Unique problems affected: {len(unique_problems)}")
                    print(f"    - Average errors per problem: {avg_errors_per_problem:.1f}")

    def print_problem_analysis(self, phase: str, phase_name: str) -> None:
        """Print analysis of which problems were affected by errors."""
        problems = self.problem_errors.get(phase, set())
        if not problems:
            return

        print(f"\nProblems with errors:")
        print(f"  • Total unique problems affected: {len(problems)}")

        # Count errors per problem
        problem_error_counts = defaultdict(int)
        all_errors = []

        # Gather all errors for this phase
        if phase == "initial":
            for errors in self.initial_errors.values():
                all_errors.extend(errors)
        elif phase == "final":
            for errors in self.final_errors.values():
                all_errors.extend(errors)
        elif phase == "between":
            for errors in self.between_errors.values():
                all_errors.extend(errors)

        for error in all_errors:
            problem_key = (error['window'], error['sample_idx'] or error['call_id'])
            if problem_key[1] is not None:
                problem_error_counts[problem_key] += 1

        if problem_error_counts:
            # Find problems with multiple errors
            multiple_errors = {k: v for k, v in problem_error_counts.items() if v > 1}
            if multiple_errors:
                print(f"  • Problems with multiple errors: {len(multiple_errors)}")
                print(f"    - Max errors for single problem: {max(multiple_errors.values())}")
                print(f"    - Average errors per problem (for problems with >1 error): {sum(multiple_errors.values()) / len(multiple_errors):.1f}")

                # Show top 5 problems with most errors
                top_problems = sorted(multiple_errors.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"    - Top problems by error count:")
                for (window, identifier), count in top_problems:
                    print(f"      * Window {window}, Sample/Call {identifier}: {count} errors")

    def print_report(self, debug=False) -> None:
        """Print a formatted report of the error analysis."""
        print("=" * 80)
        print("ERROR ANALYSIS REPORT")
        print("=" * 80)
        print(f"Log file: {self.log_file_path}")

        # Debug: Show what accuracies were captured
        if debug:
            print("\n[DEBUG] Captured accuracies:")
            for window in sorted(self.window_accuracies.keys()):
                win_acc = self.window_accuracies.get(window, "N/A")
                cum_acc = self.cumulative_accuracies.get(window, "N/A")
                samples = self.cumulative_samples.get(window, "N/A")
                print(f"  Window {window}: window_acc={win_acc}, cumulative_acc={cum_acc}, samples={samples}")
        print()

        # Initial test acc errors
        print("-" * 80)
        print("INITIAL TEST ACC CALCULATION")
        print("-" * 80)
        total_initial = sum(len(errors) for errors in self.initial_errors.values())
        print(f"Total errors: {total_initial}")

        if total_initial > 0:
            print("\nError breakdown by type:")
            for error_type, errors in sorted(self.initial_errors.items()):
                print(f"  • {error_type}: {len(errors)}")

            # Show component analysis
            self.print_component_analysis("initial", "INITIAL TEST ACC")

            # Show problem analysis
            self.print_problem_analysis("initial", "INITIAL TEST ACC")

            # Show window distribution
            window_counts = defaultdict(int)
            window_problems = defaultdict(set)  # Track unique API calls per window
            window_base_samples = defaultdict(set)  # Track unique base sample IDs per window
            for errors in self.initial_errors.values():
                for error in errors:
                    if error['window']:
                        window_counts[error['window']] += 1
                        # Track unique API calls
                        problem_id = error['sample_idx'] or error['call_id']
                        if problem_id is not None:
                            window_problems[error['window']].add(problem_id)
                            # Extract base sample ID (e.g., "1455" from "online_train_s_1455_round_2")
                            base_sample = self.extract_base_sample_id(str(problem_id))
                            if base_sample:
                                window_base_samples[error['window']].add(base_sample)
            if window_counts:
                print("\nErrors by window:")
                for window in sorted(window_counts.keys()):
                    acc_info = ""
                    if window in self.cumulative_accuracies:
                        samples_info = f", {self.cumulative_samples[window]} samples" if window in self.cumulative_samples else ""
                        acc_info = f" (cumulative acc: {self.cumulative_accuracies[window]:.3f}{samples_info})"
                    elif window in self.window_accuracies:
                        acc_info = f" (window acc: {self.window_accuracies[window]:.3f})"

                    unique_calls = len(window_problems[window])
                    unique_samples = len(window_base_samples[window])
                    # Get window size (total problems in window)
                    window_size = self.window_sizes.get(window, self.default_window_size)
                    if window_size and unique_samples > 0:
                        problems_info = f", {unique_samples}/{window_size} problems ({unique_calls} unique API calls)"
                    elif unique_samples > 0:
                        problems_info = f", {unique_samples} problems ({unique_calls} unique API calls)"
                    else:
                        problems_info = ""
                    print(f"  • Window {window}: {window_counts[window]} errors{problems_info}{acc_info}")
        print()

        # Final test acc errors
        print("-" * 80)
        print("FINAL TEST ACC CALCULATION")
        print("-" * 80)
        total_final = sum(len(errors) for errors in self.final_errors.values())
        print(f"Total errors: {total_final}")

        if total_final > 0:
            print("\nError breakdown by type:")
            for error_type, errors in sorted(self.final_errors.items()):
                print(f"  • {error_type}: {len(errors)}")

            # Show component analysis
            self.print_component_analysis("final", "FINAL TEST ACC")

            # Show problem analysis
            self.print_problem_analysis("final", "FINAL TEST ACC")

            # Show window distribution
            window_counts = defaultdict(int)
            window_problems = defaultdict(set)  # Track unique API calls per window
            window_base_samples = defaultdict(set)  # Track unique base sample IDs per window
            for errors in self.final_errors.values():
                for error in errors:
                    if error['window']:
                        window_counts[error['window']] += 1
                        # Track unique API calls
                        problem_id = error['sample_idx'] or error['call_id']
                        if problem_id is not None:
                            window_problems[error['window']].add(problem_id)
                            # Extract base sample ID (e.g., "1455" from "online_train_s_1455_round_2")
                            base_sample = self.extract_base_sample_id(str(problem_id))
                            if base_sample:
                                window_base_samples[error['window']].add(base_sample)
            if window_counts:
                print("\nErrors by window:")
                for window in sorted(window_counts.keys()):
                    acc_info = ""
                    if window in self.cumulative_accuracies:
                        samples_info = f", {self.cumulative_samples[window]} samples" if window in self.cumulative_samples else ""
                        acc_info = f" (cumulative acc: {self.cumulative_accuracies[window]:.3f}{samples_info})"
                    elif window in self.window_accuracies:
                        acc_info = f" (window acc: {self.window_accuracies[window]:.3f})"

                    unique_calls = len(window_problems[window])
                    unique_samples = len(window_base_samples[window])
                    # Get window size (total problems in window)
                    window_size = self.window_sizes.get(window, self.default_window_size)
                    if window_size and unique_samples > 0:
                        problems_info = f", {unique_samples}/{window_size} problems ({unique_calls} unique API calls)"
                    elif unique_samples > 0:
                        problems_info = f", {unique_samples} problems ({unique_calls} unique API calls)"
                    else:
                        problems_info = ""
                    print(f"  • Window {window}: {window_counts[window]} errors{problems_info}{acc_info}")
        print()

        # Between phase errors
        total_between = sum(len(errors) for errors in self.between_errors.values())
        if total_between > 0:
            print("-" * 80)
            print("ERRORS BETWEEN INITIAL AND FINAL TEST ACC")
            print("-" * 80)
            print(f"Total errors: {total_between}")
            print("\nError breakdown by type:")
            for error_type, errors in sorted(self.between_errors.items()):
                print(f"  • {error_type}: {len(errors)}")

            # Show component analysis
            self.print_component_analysis("between", "BETWEEN PHASES")

            # Show problem analysis
            self.print_problem_analysis("between", "BETWEEN PHASES")

            # Show window distribution
            window_counts = defaultdict(int)
            window_problems = defaultdict(set)  # Track unique API calls per window
            window_base_samples = defaultdict(set)  # Track unique base sample IDs per window
            for errors in self.between_errors.values():
                for error in errors:
                    if error['window']:
                        window_counts[error['window']] += 1
                        # Track unique API calls
                        problem_id = error['sample_idx'] or error['call_id']
                        if problem_id is not None:
                            window_problems[error['window']].add(problem_id)
                            # Extract base sample ID (e.g., "1455" from "online_train_s_1455_round_2")
                            base_sample = self.extract_base_sample_id(str(problem_id))
                            if base_sample:
                                window_base_samples[error['window']].add(base_sample)
            if window_counts:
                print("\nErrors by window:")
                for window in sorted(window_counts.keys()):
                    acc_info = ""
                    if window in self.cumulative_accuracies:
                        samples_info = f", {self.cumulative_samples[window]} samples" if window in self.cumulative_samples else ""
                        acc_info = f" (cumulative acc: {self.cumulative_accuracies[window]:.3f}{samples_info})"
                    elif window in self.window_accuracies:
                        acc_info = f" (window acc: {self.window_accuracies[window]:.3f})"

                    unique_calls = len(window_problems[window])
                    unique_samples = len(window_base_samples[window])
                    # Get window size (total problems in window)
                    window_size = self.window_sizes.get(window, self.default_window_size)
                    if window_size and unique_samples > 0:
                        problems_info = f", {unique_samples}/{window_size} problems ({unique_calls} unique API calls)"
                    elif unique_samples > 0:
                        problems_info = f", {unique_samples} problems ({unique_calls} unique API calls)"
                    else:
                        problems_info = ""
                    print(f"  • Window {window}: {window_counts[window]} errors{problems_info}{acc_info}")
            print()

        # Detailed error samples
        if total_initial > 0 or total_final > 0 or total_between > 0:
            print("=" * 80)
            print("DETAILED ERROR SAMPLES")
            print("=" * 80)

            if total_initial > 0:
                print("\n[INITIAL TEST ACC - Sample errors]")
                for error_type, errors in sorted(self.initial_errors.items()):
                    window_info = f"Window {errors[0]['window']}" if errors[0]['window'] else "Window unknown"
                    print(f"\n  {error_type} (showing first of {len(errors)}):")
                    print(f"    {window_info}, Line {errors[0]['line_num']}: {errors[0]['message'][:100]}...")
                    if errors[0]['details']:
                        print(f"    Details: {errors[0]['details'][:200]}...")

            if total_final > 0:
                print("\n[FINAL TEST ACC - Sample errors]")
                for error_type, errors in sorted(self.final_errors.items()):
                    window_info = f"Window {errors[0]['window']}" if errors[0]['window'] else "Window unknown"
                    print(f"\n  {error_type} (showing first of {len(errors)}):")
                    print(f"    {window_info}, Line {errors[0]['line_num']}: {errors[0]['message'][:100]}...")
                    if errors[0]['details']:
                        print(f"    Details: {errors[0]['details'][:200]}...")

            if total_between > 0:
                print("\n[BETWEEN PHASES - Sample errors]")
                for error_type, errors in sorted(self.between_errors.items()):
                    window_info = f"Window {errors[0]['window']}" if errors[0]['window'] else "Window unknown"
                    print(f"\n  {error_type} (showing first of {len(errors)}):")
                    print(f"    {window_info}, Line {errors[0]['line_num']}: {errors[0]['message'][:100]}...")
                    if errors[0]['details']:
                        print(f"    Details: {errors[0]['details'][:200]}...")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Initial test acc errors:  {total_initial}")
        print(f"Final test acc errors:    {total_final}")
        print(f"Between phases errors:    {total_between}")
        print(f"Total errors:             {total_initial + total_final + total_between}")

        # Special callout for specific windows
        special_windows = [30, 50, 70, 80, 90, 100]
        windows_found = [w for w in special_windows if w in self.window_accuracies]

        if windows_found:
            print("\nKey Window Statistics:")
            for window_num in windows_found:
                # Count errors in this window
                window_errors = 0
                for errors in self.initial_errors.values():
                    window_errors += sum(1 for e in errors if e['window'] == window_num)
                for errors in self.final_errors.values():
                    window_errors += sum(1 for e in errors if e['window'] == window_num)
                for errors in self.between_errors.values():
                    window_errors += sum(1 for e in errors if e['window'] == window_num)

                # Prefer cumulative accuracy over window accuracy
                if window_num in self.cumulative_accuracies:
                    acc_value = self.cumulative_accuracies[window_num]
                    acc_label = "Cumulative Acc"
                    samples = self.cumulative_samples.get(window_num)
                    samples_str = f" ({samples} samples)" if samples else ""
                elif window_num in self.window_accuracies:
                    acc_value = self.window_accuracies[window_num]
                    acc_label = "Window Acc"
                    samples_str = ""
                else:
                    acc_value = None
                    acc_label = "Accuracy"
                    samples_str = ""

                if acc_value is not None:
                    print(f"  Window {window_num:3d} - {acc_label}: {acc_value:.3f}{samples_str}, Errors: {window_errors}")
                else:
                    print(f"  Window {window_num:3d} - Errors: {window_errors}")

        print("=" * 80)


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_errors.py <path_to_terminal_log_file> [--debug]")
        print("\nExample:")
        print("  python analyze_errors.py results/ace_run/terminal_log.txt")
        print("  python analyze_errors.py results/ace_run_20240115/terminal_log.txt")
        print("  python analyze_errors.py results/ace_run/terminal_log.txt --debug")
        sys.exit(1)

    log_file = sys.argv[1]
    debug = "--debug" in sys.argv

    analyzer = ErrorAnalyzer(log_file)
    analyzer.analyze_log()
    analyzer.print_report(debug=debug)


if __name__ == "__main__":
    main()
