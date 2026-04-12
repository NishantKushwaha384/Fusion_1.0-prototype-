# ============================================================
# FUSION 1.0 — QUERY LOGGER
# ============================================================
# This file saves EVERY query and response to a log file.
#
# Why logging matters for YOUR project:
#   Every logged entry is a data point for your research.
#   After 100 entries, you can:
#   - Find which models perform best per category
#   - Find where fusion helps vs hurts
#   - Identify your system's failure modes
#   - Build your benchmark dataset
#   This log is your most valuable research asset.
#
# Log format: JSONL (one JSON object per line)
# Easy to read with pandas later:
#   import pandas as pd
#   df = pd.read_json("logger.txt", lines=True)
# ============================================================

import json
import os
from datetime import datetime


# Path where logs are stored
LOG_FILE = "logger.txt"


def load_logs_as_list() -> list:
    """
    Load all logs from the JSONL file as a list of dictionaries.

    Returns:
        List of log entries, each as a dict

    Example:
        logs = load_logs_as_list()
        print(f"Total logs: {len(logs)}")
    """
    if not os.path.exists(LOG_FILE):
        return []

    logs = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
    except Exception as e:
        print(f"[LOGGER] Error loading logs: {e}")
        return []

    return logs


def log_query(
    question: str,
    classification: dict,
    dispatch_result: dict,
    fusion_result: dict,
    latency: float
) -> None:
    """
    Saves a complete query record to the log file.
    Called automatically from main.py after every query.

    Each log entry contains:
    - timestamp:          When the query happened
    - question:           The original user question
    - category:           What type of question (math, coding etc)
    - complexity:         low / medium / high
    - models_used:        Which models answered
    - individual_answers: Raw answer from each model
    - confidence_scores:  Each model's self-reported confidence
    - fusion_weights:     How much each model contributed
    - final_answer:       The fused output
    - strategy:           Which fusion strategy was applied
    - latency:            Total time in seconds
    - fusion_notes:       Notes from the fusion engine
    """

    log_entry = {
        # ── META ─────────────────────────────────────────────
        "timestamp":   datetime.now().isoformat(),
        "session_id":  _get_session_id(),

        # ── QUESTION ─────────────────────────────────────────
        "question":    question,
        "char_count":  len(question),

        # ── CLASSIFICATION ───────────────────────────────────
        "category":         classification.get("category"),
        "complexity":        classification.get("complexity"),
        "classifier_confidence": classification.get("confidence"),
        "classifier_reasoning":  classification.get("reasoning"),

        # ── DISPATCH ─────────────────────────────────────────
        "models_used":       dispatch_result.get("models_used", []),
        "individual_answers": dispatch_result.get("answers", []),
        "confidence_scores": dispatch_result.get("confidence_scores", []),
        "model_latencies":   dispatch_result.get("latencies", []),
        "strategy":          dispatch_result.get("strategy"),

        # ── FUSION ───────────────────────────────────────────
        "final_answer":   fusion_result.get("answer"),
        "fusion_weights": fusion_result.get("weights", []),
        "fusion_strategy": fusion_result.get("strategy"),
        "fusion_notes":   fusion_result.get("notes"),

        # ── PERFORMANCE ──────────────────────────────────────
        "total_latency_seconds": latency,

        # ── YOUR RESEARCH FIELDS ─────────────────────────────
        # Add fields here as you discover what you need to track.
        # For example, when you implement quality scoring:
        # "fusion_quality_score": None,    # Fill in later
        # "human_rating": None,            # Fill in during evaluation
        # "better_than_single": None,      # Fill in during benchmark
    }

    # Append to log file (one JSON per line = JSONL format)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"[LOGGER] Logged query #{_count_logs()} → {LOG_FILE}")


def _get_session_id() -> str:
    """
    Returns today's date as a session ID.
    Groups logs by day for easy analysis.
    """
    return datetime.now().strftime("%Y-%m-%d")


def _count_logs() -> int:
    """Returns total number of logged queries."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


# ── ANALYSIS UTILITIES ───────────────────────────────────────
# Helper functions for studying your logs during research.
# Run these in a Jupyter notebook or separate analysis script.

def search_logs_by_question(search_term: str, case_sensitive: bool = False) -> list:
    """
    Search logs by question content.
    Returns all logs where the question contains the search term.

    Args:
        search_term: Text to search for in questions
        case_sensitive: Whether search should be case-sensitive

    Example:
        from logger import search_logs_by_question
        math_logs = search_logs_by_question("derivative")
        print(f"Found {len(math_logs)} questions about derivatives")
    """
    logs = load_logs_as_list()
    if not logs:
        return []

    if not case_sensitive:
        search_term = search_term.lower()

    matches = []
    for log in logs:
        question = log.get("question", "")
        if not case_sensitive:
            question = question.lower()

        if search_term in question:
            matches.append(log)

    return matches


def get_logs_by_category(category: str) -> list:
    """
    Get all logs for a specific category.

    Args:
        category: Category name (math, coding, factual, etc.)

    Example:
        from logger import get_logs_by_category
        math_logs = get_logs_by_category("math")
        print(f"Total math questions: {len(math_logs)}")
    """
    logs = load_logs_as_list()
    return [log for log in logs if log.get("category") == category]


def get_recent_logs(limit: int = 10) -> list:
    """
    Get the most recent logs.

    Args:
        limit: Number of recent logs to return (default: 10)

    Example:
        from logger import get_recent_logs
        recent = get_recent_logs(5)
        for log in recent:
            print(f"{log['timestamp']}: {log['question'][:50]}...")
    """
    logs = load_logs_as_list()
    # Sort by timestamp (most recent first)
    sorted_logs = sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
    return sorted_logs[:limit]


def get_logs_by_date_range(start_date: str, end_date: str = None) -> list:
    """
    Get logs within a date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional, defaults to today)

    Example:
        from logger import get_logs_by_date_range
        week_logs = get_logs_by_date_range("2024-01-01", "2024-01-07")
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logs = load_logs_as_list()
    filtered_logs = []

    for log in logs:
        timestamp = log.get("timestamp", "")
        if timestamp:
            log_date = timestamp.split("T")[0]  # Extract date part
            if start_date <= log_date <= end_date:
                filtered_logs.append(log)

    return filtered_logs


def get_log_details(log_index: int = -1) -> dict:
    """
    Get detailed information about a specific log entry.

    Args:
        log_index: Index of the log entry (default: -1 for most recent)

    Example:
        from logger import get_log_details
        latest = get_log_details()  # Most recent log
        print(f"Question: {latest['question']}")
        print(f"Answer: {latest['final_answer']}")
        print(f"Latency: {latest['total_latency_seconds']:.2f}s")
    """
    logs = load_logs_as_list()
    if not logs:
        return {}

    if log_index < 0:
        log_index = len(logs) + log_index  # Handle negative indexing

    if 0 <= log_index < len(logs):
        return logs[log_index]
    else:
        print(f"[LOGGER] Invalid log index: {log_index}. Total logs: {len(logs)}")
        return {}


def export_logs_to_csv(filename: str = "fusion_logs.csv") -> None:
    """
    Export logs to CSV format for analysis in Excel/spreadsheets.

    Args:
        filename: Output filename (default: fusion_logs.csv)

    Example:
        from logger import export_logs_to_csv
        export_logs_to_csv("my_analysis.csv")
    """
    import csv

    logs = load_logs_as_list()
    if not logs:
        print("[LOGGER] No logs to export")
        return

    # Flatten nested data for CSV
    flattened_logs = []
    for log in logs:
        flat_log = {
            "timestamp": log.get("timestamp"),
            "question": log.get("question"),
            "category": log.get("category"),
            "complexity": log.get("complexity"),
            "models_used": ", ".join(log.get("models_used", [])),
            "confidence_scores": ", ".join([f"{c:.2f}" for c in log.get("confidence_scores", [])]),
            "model_latencies": ", ".join([f"{l:.2f}" for l in log.get("model_latencies", [])]),
            "strategy": log.get("strategy"),
            "fusion_strategy": log.get("fusion_strategy"),
            "total_latency_seconds": log.get("total_latency_seconds"),
            "final_answer": log.get("final_answer", "")[:500]  # Truncate long answers
        }
        flattened_logs.append(flat_log)

    if flattened_logs:
        fieldnames = flattened_logs[0].keys()
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_logs)

        print(f"[LOGGER] Exported {len(flattened_logs)} logs to {filename}")


def get_performance_stats() -> dict:
    """
    Get detailed performance statistics from all logs.

    Returns:
        dict with various performance metrics

    Example:
        from logger import get_performance_stats
        stats = get_performance_stats()
        print(f"Average latency: {stats['avg_latency']:.2f}s")
        print(f"Most common category: {stats['top_category']}")
    """
    logs = load_logs_as_list()
    if not logs:
        return {}

    from collections import Counter

    # Basic stats
    total_queries = len(logs)
    categories = Counter(log.get("category") for log in logs if log.get("category"))
    complexities = Counter(log.get("complexity") for log in logs if log.get("complexity"))

    # Latency stats
    latencies = [log.get("total_latency_seconds", 0) for log in logs if log.get("total_latency_seconds")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    # Confidence stats
    all_confidences = []
    for log in logs:
        all_confidences.extend(log.get("confidence_scores", []))
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    # Model usage stats
    model_usage = Counter()
    for log in logs:
        for model in log.get("models_used", []):
            model_usage[model] += 1

    return {
        "total_queries": total_queries,
        "avg_latency": round(avg_latency, 2),
        "min_latency": round(min_latency, 2),
        "max_latency": round(max_latency, 2),
        "avg_confidence": round(avg_confidence, 2),
        "categories": dict(categories.most_common()),
        "complexities": dict(complexities.most_common()),
        "model_usage": dict(model_usage.most_common()),
        "top_category": categories.most_common(1)[0][0] if categories else None,
        "top_model": model_usage.most_common(1)[0][0] if model_usage else None
    }


def search_logs_by_answer(search_term: str, case_sensitive: bool = False) -> list:
    """
    Search logs by answer content.
    Returns all logs where the final answer contains the search term.

    Args:
        search_term: Text to search for in answers
        case_sensitive: Whether search should be case-sensitive

    Example:
        from logger import search_logs_by_answer
        python_logs = search_logs_by_answer("def ")
        print(f"Found {len(python_logs)} answers with Python code")
    """
    logs = load_logs_as_list()
    if not logs:
        return []

    if not case_sensitive:
        search_term = search_term.lower()

    matches = []
    for log in logs:
        answer = log.get("final_answer", "")
        if not case_sensitive:
            answer = answer.lower()

        if search_term in answer:
            matches.append(log)

    return matches


def get_logs_by_performance(min_confidence: float = None, max_latency: float = None) -> list:
    """
    Get logs filtered by performance criteria.

    Args:
        min_confidence: Minimum average confidence score (0.0-1.0)
        max_latency: Maximum latency in seconds

    Example:
        from logger import get_logs_by_performance
        fast_logs = get_logs_by_performance(max_latency=5.0)
        confident_logs = get_logs_by_performance(min_confidence=0.8)
    """
    logs = load_logs_as_list()
    filtered_logs = []

    for log in logs:
        include_log = True

        # Check confidence filter
        if min_confidence is not None:
            confidences = log.get("confidence_scores", [])
            if confidences:
                avg_conf = sum(confidences) / len(confidences)
                if avg_conf < min_confidence:
                    include_log = False

        # Check latency filter
        if max_latency is not None:
            latency = log.get("total_latency_seconds", 0)
            if latency > max_latency:
                include_log = False

        if include_log:
            filtered_logs.append(log)

    return filtered_logs


def backup_logs(backup_filename: str = None) -> str:
    """
    Create a backup of the current log file.

    Args:
        backup_filename: Custom backup filename (optional)

    Returns:
        Path to the backup file created

    Example:
        from logger import backup_logs
        backup_path = backup_logs("fusion_logs_backup_2024.jsonl")
    """
    if backup_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"fusion_logs_backup_{timestamp}.jsonl"

    try:
        # Copy the log file
        with open(LOG_FILE, "r", encoding="utf-8") as src:
            with open(backup_filename, "w", encoding="utf-8") as dst:
                dst.write(src.read())

        print(f"[LOGGER] Backup created: {backup_filename}")
        return backup_filename
    except FileNotFoundError:
        print(f"[LOGGER] No log file to backup: {LOG_FILE}")
        return ""
    except Exception as e:
        print(f"[LOGGER] Backup failed: {e}")
        return ""


def get_log_by_id(log_id: int) -> dict:
    """
    Get a specific log entry by its sequential ID.

    Args:
        log_id: The sequential ID of the log (1-based, first log = 1)

    Example:
        from logger import get_log_by_id
        log = get_log_by_id(42)  # Get the 42nd logged query
        if log:
            print(f"Question: {log['question']}")
    """
    logs = load_logs_as_list()
    if 1 <= log_id <= len(logs):
        return logs[log_id - 1]  # Convert to 0-based indexing
    else:
        print(f"[LOGGER] Invalid log ID: {log_id}. Valid range: 1-{len(logs)}")
        return {}


def print_log_details(log_id: int = -1) -> None:
    """
    Print detailed information about a specific log entry.

    Args:
        log_id: Log ID to display (default: -1 for most recent)

    Example:
        from logger import print_log_details
        print_log_details(5)  # Show details of the 5th logged query
    """
    log = get_log_details(log_id)
    if not log:
        return

    print("\n" + "="*80)
    print(f"LOG ENTRY #{log_id if log_id > 0 else 'MOST RECENT'}")
    print("="*80)
    print(f"Timestamp: {log.get('timestamp')}")
    print(f"Session: {log.get('session_id')}")
    print(f"Category: {log.get('category')} | Complexity: {log.get('complexity')}")
    print(f"Total Latency: {log.get('total_latency_seconds', 0):.2f}s")
    print(f"Strategy: {log.get('strategy')} → {log.get('fusion_strategy')}")
    print(f"\nQUESTION ({log.get('char_count')} chars):")
    print(f"  {log.get('question')}")
    print(f"\nMODELS USED: {', '.join(log.get('models_used', []))}")
    print(f"CONFIDENCES: {', '.join([f'{c:.2f}' for c in log.get('confidence_scores', [])])}")
    print(f"LATENCIES: {', '.join([f'{l:.2f}s' for l in log.get('model_latencies', [])])}")
    print(f"\nFINAL ANSWER:")
    answer = log.get('final_answer', '')
    # Print answer with line wrapping for readability
    for i in range(0, len(answer), 100):
        print(f"  {answer[i:i+100]}")
    print("="*80 + "\n")
    """
    Loads all logs as a Python list of dicts.
    Use for quick analysis in Python scripts.

    Example usage:
        from logger import load_logs_as_list
        logs = load_logs_as_list()
        math_logs = [l for l in logs if l["category"] == "math"]
        avg_conf = sum(l["classifier_confidence"] for l in math_logs) / len(math_logs)
    """
    logs = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
    except FileNotFoundError:
        print(f"[LOGGER] No log file found at {LOG_FILE}")
    return logs


def print_summary() -> None:
    """
    Prints a summary of all logged queries.
    Run this from terminal to quickly see system performance:
        python -c "from logger import print_summary; print_summary()"
    """
    logs = load_logs_as_list()
    if not logs:
        print("No logs yet. Run some queries first.")
        return

    from collections import defaultdict, Counter

    total = len(logs)
    categories = Counter(l.get("category") for l in logs)
    avg_latency = sum(l.get("total_latency_seconds", 0) for l in logs) / total

    print("\n" + "="*50)
    print(f"  FUSION 1.0 — LOG SUMMARY ({total} queries)")
    print("="*50)
    print(f"  Average latency: {avg_latency:.2f}s")
    print(f"\n  Queries by category:")
    for cat, count in categories.most_common():
        print(f"    {cat:<12} : {count} queries")
    print("="*50 + "\n")


def get_logs_by_model(model_name: str) -> list:
    """
    Get all logs where a specific model was used.

    Args:
        model_name: Name of the model (groq, gemini, ollama)

    Example:
        from logger import get_logs_by_model
        groq_logs = get_logs_by_model("groq")
        print(f"Groq was used in {len(groq_logs)} queries")
    """
    logs = load_logs_as_list()
    return [log for log in logs if model_name in log.get("models_used", [])]


def analyze_fusion_effectiveness() -> dict:
    """
    Analyze how well fusion is working compared to individual models.

    Returns:
        dict with fusion effectiveness metrics

    Example:
        from logger import analyze_fusion_effectiveness
        analysis = analyze_fusion_effectiveness()
        print(f"Fusion confidence advantage: {analysis['avg_confidence_gain']:.2f}")
    """
    logs = load_logs_as_list()
    if not logs:
        return {}

    total_logs = len(logs)
    fusion_confidences = []
    individual_avg_confidences = []

    for log in logs:
        confidences = log.get("confidence_scores", [])
        if confidences:
            # Average confidence of individual models
            individual_avg = sum(confidences) / len(confidences)
            individual_avg_confidences.append(individual_avg)

            # For fusion effectiveness, we'll assume fusion confidence is the max individual confidence
            # (since we don't have a direct fusion confidence score yet)
            fusion_confidences.append(max(confidences))

    if not fusion_confidences or not individual_avg_confidences:
        return {}

    avg_fusion_conf = sum(fusion_confidences) / len(fusion_confidences)
    avg_individual_conf = sum(individual_avg_confidences) / len(individual_avg_confidences)
    confidence_gain = avg_fusion_conf - avg_individual_conf

    return {
        "total_analyzed": total_logs,
        "avg_fusion_confidence": round(avg_fusion_conf, 3),
        "avg_individual_confidence": round(avg_individual_conf, 3),
        "avg_confidence_gain": round(confidence_gain, 3),
        "fusion_better_percentage": round((confidence_gain > 0) * 100, 1)
    }


def clear_logs(confirm: bool = False) -> bool:
    """
    Clear all logged data. USE WITH CAUTION!

    Args:
        confirm: Must be True to actually clear logs

    Returns:
        True if logs were cleared, False otherwise

    Example:
        from logger import clear_logs
        # This will NOT clear logs:
        clear_logs()
        # This WILL clear logs:
        clear_logs(confirm=True)
    """
    if not confirm:
        print("[LOGGER] Clear operation not confirmed. Set confirm=True to clear all logs.")
        print("        This action cannot be undone!")
        return False

    try:
        # Create backup first
        backup_name = backup_logs()
        if backup_name:
            print(f"[LOGGER] Backup created: {backup_name}")

        # Clear the log file
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")

        print(f"[LOGGER] All logs cleared from {LOG_FILE}")
        return True
    except Exception as e:
        print(f"[LOGGER] Failed to clear logs: {e}")
        return False


def get_random_sample(sample_size: int = 5) -> list:
    """
    Get a random sample of logs for analysis.

    Args:
        sample_size: Number of random logs to return

    Example:
        from logger import get_random_sample
        sample = get_random_sample(10)
        for log in sample:
            print(f"Category: {log['category']}, Question: {log['question'][:50]}...")
    """
    import random

    logs = load_logs_as_list()
    if len(logs) <= sample_size:
        return logs

    return random.sample(logs, sample_size)


# ── COMMAND LINE INTERFACE ────────────────────────────────────

def main():
    """
    Command-line interface for the logger.
    Run: python logger.py <command> [args...]
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python logger.py <command>")
        print("Commands:")
        print("  summary     - Show log summary")
        print("  details     - Show details of most recent log")
        print("  details <n> - Show details of log #n")
        print("  search <term> - Search questions for term")
        print("  category <cat> - Show logs for category")
        print("  export      - Export logs to CSV")
        print("  backup      - Create log backup")
        print("  stats       - Show performance statistics")
        print("  sample <n>  - Show random sample of n logs")
        return

    command = sys.argv[1].lower()

    try:
        if command == "summary":
            print_summary()
        elif command == "details":
            log_id = int(sys.argv[2]) if len(sys.argv) > 2 else -1
            print_log_details(log_id)
        elif command == "search":
            if len(sys.argv) < 3:
                print("Usage: python logger.py search <search_term>")
                return
            results = search_logs_by_question(sys.argv[2])
            print(f"Found {len(results)} matching questions")
            for i, log in enumerate(results[:5]):  # Show first 5
                print(f"{i+1}. {log['question'][:80]}...")
        elif command == "category":
            if len(sys.argv) < 3:
                print("Usage: python logger.py category <category_name>")
                return
            results = get_logs_by_category(sys.argv[2])
            print(f"Found {len(results)} {sys.argv[2]} questions")
        elif command == "export":
            filename = sys.argv[2] if len(sys.argv) > 2 else "fusion_logs.csv"
            export_logs_to_csv(filename)
        elif command == "backup":
            backup_logs()
        elif command == "stats":
            stats = get_performance_stats()
            print(f"Total queries: {stats.get('total_queries', 0)}")
            print(f"Average latency: {stats.get('avg_latency', 0):.2f}s")
            print(f"Average confidence: {stats.get('avg_confidence', 0):.2f}")
        elif command == "sample":
            sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            sample = get_random_sample(sample_size)
            print(f"Random sample of {len(sample)} logs:")
            for i, log in enumerate(sample):
                print(f"{i+1}. [{log.get('category')}] {log['question'][:60]}...")
        else:
            print(f"Unknown command: {command}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
