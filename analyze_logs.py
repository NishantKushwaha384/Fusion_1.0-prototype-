#!/usr/bin/env python3
"""
Analyze logs with pandas
"""

import pandas as pd
import os

def main():
    if not os.path.exists('logger.txt'):
        print('📝 logger.txt not found yet. Run some queries first!')
        return

    print('📊 Log Analysis with Pandas:')
    print('=' * 40)

    # Load logs into pandas DataFrame
    df = pd.read_json('logger.txt', lines=True)
    print(f'📈 Total queries: {len(df)}')

    if len(df) > 0:
        # Category breakdown
        print(f'📂 Categories: {df["category"].value_counts().to_dict()}')

        # Performance metrics
        print(f'⚡ Average latency: {df["total_latency_seconds"].mean():.2f}s')
        print(f'🎯 Average confidence: {df["confidence_scores"].apply(lambda x: sum(x)/len(x) if x else 0).mean():.2f}')

        # Sample questions
        print('💬 Sample questions:')
        for i, row in df.head(3).iterrows():
            print(f'   {i+1}. [{row["category"]}] {row["question"][:50]}...')

        # Export to CSV option
        csv_file = 'logs_analysis.csv'
        df.to_csv(csv_file, index=False)
        print(f'💾 Exported analysis to: {csv_file}')

if __name__ == "__main__":
    main()