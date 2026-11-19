#!/usr/bin/env python3
"""
Overnight Automated Submission Dispatcher for PGS-S6E11.
Executes submissions for Candidate 1 (Sovereign 5D Apex) and Candidate 2 (Step I Pure CV),
waits for scoring, and logs the resulting Public Leaderboard scores and ranks.
"""

import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

KAGGLE_BIN = '/home/shivamgravity/miniconda3/envs/kaggle_base/bin/kaggle'
COMPETITION = 'playground-series-s6e11'
SUB_DIR = Path('pgs-s6e11/submission')
LOG_FILE = Path('pgs-s6e11/submission_overnight.log')

def log(msg):
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip(), res.stderr.strip(), res.returncode

def submit_file(file_path, description):
    log(f"Submitting {file_path.name}...")
    cmd = f'{KAGGLE_BIN} competitions submit -c {COMPETITION} -f {file_path} -m "{description}"'
    stdout, stderr, code = run_cmd(cmd)
    log(f"Submit output: {stdout}")
    if stderr:
        log(f"Submit stderr: {stderr}")
    return code == 0

def check_latest_submission():
    cmd = f'{KAGGLE_BIN} competitions submissions -c {COMPETITION}'
    stdout, stderr, code = run_cmd(cmd)
    if code != 0:
        log(f"Error checking submissions: {stderr}")
        return None
    lines = stdout.split('\n')
    header_idx = -1
    for i, line in enumerate(lines):
        if 'ref' in line and 'fileName' in line:
            header_idx = i
            break
    if header_idx != -1 and len(lines) > header_idx + 2:
        top_sub = lines[header_idx + 2]
        log(f"Latest submission record: {top_sub}")
        return top_sub
    return None

def main():
    log("=" * 80)
    log("🚀 Starting PGS-S6E11 Midnight Quota Automated Submission Run")
    log("=" * 80)

    cand1 = SUB_DIR / 'sovereign_5d_gram_schmidt_apex_submission.csv'
    cand2 = SUB_DIR / 'sovereign_step_i_orthogonal_apex_submission.csv'

    assert cand1.exists(), f"Missing {cand1}"
    assert cand2.exists(), f"Missing {cand2}"

    # Submit Candidate 1: Frontier Summit Apex
    desc1 = "Sovereign 5D Gram-Schmidt Apex (Step J + Ortho TabM PWL + BPE LGB + RealMLP + GPT2 + 4-Zone Manifold)"
    ok1 = submit_file(cand1, desc1)
    
    if ok1:
        log("Candidate 1 submitted. Waiting 60 seconds for Kaggle to score...")
        time.sleep(60)
        check_latest_submission()

    log("-" * 60)
    log("Proceeding to submit Candidate 2: Pure CV Foundation...")
    desc2 = "Sovereign Step I Orthogonal Apex (Step I Pure CV + Ortho TabM PWL + BPE LGB + RealMLP + GPT2 + 4-Zone Manifold)"
    ok2 = submit_file(cand2, desc2)

    if ok2:
        log("Candidate 2 submitted. Waiting 60 seconds for Kaggle to score...")
        time.sleep(60)
        check_latest_submission()

    log("=" * 80)
    log("✅ Midnight Submission Run Completed Successfully!")
    log("=" * 80)

if __name__ == '__main__':
    main()
