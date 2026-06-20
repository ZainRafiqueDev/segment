# backend/1_clean_data.py
# DAG Node 1 — Raw Data Cleaning
# Run: python 1_clean_data.py
# EXPLAIN: "I ran a cleaning script that checked all 50,000 reviews for duplicates,
#           HTML artifacts, empty content, and extreme lengths. I fixed HTML in-place
#           and flagged short/long outliers. This ensures the training set has clean
#           signal before tokenization."

import os, re, json
import pandas as pd
from tqdm import tqdm

RAW_FILE   = 'IMDB Dataset.csv'
CLEAN_FILE = 'imdb_clean.csv'
MAX_WORDS  = 500
MIN_WORDS  = 10

def clean_review(text):
    """Remove HTML, normalize whitespace, fix common encoding artifacts."""
    text = re.sub(r'<[^>]+>', ' ', str(text))   # strip HTML tags
    text = re.sub(r'&amp;',  '&',  text)
    text = re.sub(r'&lt;',   '<',  text)
    text = re.sub(r'&gt;',   '>',  text)
    text = re.sub(r'&quot;', '"',  text)
    text = re.sub(r'&#\d+;', ' ',  text)
    text = re.sub(r'\s+',    ' ',  text).strip()
    return text

print('=== SENTIMENTSAGE DATA CLEANING ===\n')

if not os.path.exists(RAW_FILE):
    print(f'ERROR: {RAW_FILE} not found.')
    print('Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews')
    exit(1)

df = pd.read_csv(RAW_FILE)
original_count = len(df)
print(f'Raw dataset:  {original_count:,} reviews')
print(f'Columns:      {df.columns.tolist()}')
print(f'Labels:       {df["sentiment"].value_counts().to_dict()}')

issues = {}

# ── Check 1: Duplicates ────────────────────────────────────────────────
dupes = df.duplicated(subset=['review']).sum()
issues['duplicates_removed'] = int(dupes)
df = df.drop_duplicates(subset=['review']).reset_index(drop=True)

# ── Check 2: Nulls / empty ────────────────────────────────────────────
nulls = int(df['review'].isnull().sum())
df['review'] = df['review'].fillna('')
empty = int((df['review'].str.strip() == '').sum())
issues['null_removed']  = nulls
issues['empty_removed'] = empty
df = df[df['review'].str.strip() != ''].reset_index(drop=True)

# ── Check 3: HTML artifacts — fix in-place ────────────────────────────
html_mask = df['review'].str.contains(r'<[^>]+>', regex=True)
issues['html_fixed'] = int(html_mask.sum())
df['review'] = df['review'].apply(clean_review)

# ── Check 4: Very short reviews ───────────────────────────────────────
word_counts = df['review'].str.split().str.len()
short_mask  = word_counts < MIN_WORDS
issues['short_removed'] = int(short_mask.sum())
df = df[~short_mask].reset_index(drop=True)

# ── Check 5: Very long reviews — truncate ─────────────────────────────
word_counts = df['review'].str.split().str.len()
long_mask   = word_counts > MAX_WORDS
issues['long_truncated'] = int(long_mask.sum())
df['review'] = df['review'].apply(
    lambda x: ' '.join(x.split()[:MAX_WORDS]) if len(x.split()) > MAX_WORDS else x
)

# ── Save ──────────────────────────────────────────────────────────────
df.to_csv(CLEAN_FILE, index=False)

# ── Report ────────────────────────────────────────────────────────────
total_removed = issues['duplicates_removed'] + issues['null_removed'] + issues['empty_removed'] + issues['short_removed']

print(f'\n--- CLEANING REPORT ---')
print(f'Duplicates removed:     {issues["duplicates_removed"]}')
print(f'Null reviews removed:   {issues["null_removed"]}')
print(f'Empty reviews removed:  {issues["empty_removed"]}')
print(f'HTML artifacts fixed:   {issues["html_fixed"]} (fixed in-place, not removed)')
print(f'Short reviews removed:  {issues["short_removed"]} (< {MIN_WORDS} words)')
print(f'Long reviews truncated: {issues["long_truncated"]} (capped at {MAX_WORDS} words)')
print(f'\nOriginal count:  {original_count:,}')
print(f'Final count:     {len(df):,}')
print(f'Total removed:   {total_removed}')
print(f'\nFinal label distribution:')
print(df['sentiment'].value_counts().to_dict())
print(f'\nSaved to: {CLEAN_FILE}')

with open('cleaning_report.json', 'w') as f:
    json.dump({**issues, 'original': original_count, 'final': len(df)}, f)
print('Report saved to: cleaning_report.json')
