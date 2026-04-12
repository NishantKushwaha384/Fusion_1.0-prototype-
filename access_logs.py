#!/usr/bin/env python3
"""
Quick script to access logged queries from logger.txt
"""

from logger import load_logs_as_list, get_performance_stats

def main():
    print("🔍 Accessing logged queries from logger.txt")
    print("=" * 50)

    # Load all logs
    logs = load_logs_as_list()
    print(f"📊 Total logged queries: {len(logs)}")

    if not logs:
        print("❌ No logs yet - run some queries first!")
        print("\n💡 To generate logs:")
        print("   1. Start the server: python main.py")
        print("   2. Visit: http://localhost:8000")
        print("   3. Ask a question")
        return

    # Show performance stats
    stats = get_performance_stats()
    print(f"⚡ Average latency: {stats['avg_latency']:.2f}s")
    print(f"🎯 Average confidence: {stats['avg_confidence']:.2f}")
    print(f"📂 Categories: {list(stats['categories'].keys())}")

    # Show most recent query
    recent = logs[-1]
    print(f"\n🕒 Most recent query:")
    print(f"   Question: {recent['question'][:60]}...")
    print(f"   Category: {recent['category']}")
    print(f"   Answer length: {len(recent['final_answer'])} chars")

if __name__ == "__main__":
    main()