#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the full audit workflow
"""
import subprocess
import sys
import os

os.chdir(r"D:\2026\KTNN\Research audit bank\new source vscode 1.1")

print("="*80)
print("RUNNING FULL AUDIT WORKFLOW (0_example_usage.py)")
print("="*80)
print()

# Run the main workflow
result = subprocess.run(
    [sys.executable, "0_example_usage.py"],
    capture_output=True,
    text=True,
    timeout=300
)

# Print last 100 lines of output
lines = result.stdout.split('\n')
print('\n'.join(lines[-100:]))

if result.stderr:
    print("\n" + "="*80)
    print("STDERR (last 50 lines):")
    print("="*80)
    stderr_lines = result.stderr.split('\n')
    print('\n'.join(stderr_lines[-50:]))

print("\n" + "="*80)
print("OUTPUTS FOLDER CONTENTS AFTER RUN:")
print("="*80)

excel_files = []
csv_files = []
json_files = []
other_files = []

for fname in sorted(os.listdir("outputs")):
    fpath = os.path.join("outputs", fname)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        if fname.lower().endswith(".xlsx"):
            excel_files.append((fname, size))
        elif fname.lower().endswith(".csv"):
            csv_files.append((fname, size))
        elif fname.lower().endswith(".json"):
            json_files.append((fname, size))
        else:
            other_files.append((fname, size))

print(f"\nExcel Files: ({len(excel_files)})")
for fname, size in excel_files:
    print(f"  [EXCEL] {fname:<40} ({size:>10,} bytes)")

print(f"\nCSV Files: ({len(csv_files)})")
if csv_files:
    print("  [WARNING] Should only have 'all_banks_summary.csv':")
    for fname, size in csv_files:
        marker = "[OK]" if fname == "all_banks_summary.csv" else "[REMOVE]"
        print(f"  {marker} {fname:<40} ({size:>10,} bytes)")
else:
    print("  (none)")

print(f"\nJSON Files: ({len(json_files)}) - Bank reports")
if json_files:
    for fname, size in json_files[:3]:
        print(f"  {fname:<40} ({size:>10,} bytes)")
    if len(json_files) > 3:
        print(f"  ... and {len(json_files) - 3} more bank reports")
else:
    print("  (none)")

print(f"\nOther Files: ({len(other_files)}) - Images, dashboards, etc.")
if other_files and other_files[0][0].endswith(".png"):
    print(f"  {other_files[0][0]:<40} (dashboard images)")
    print(f"  ... and {len(other_files) - 1} more dashboard files")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
if len(excel_files) == 1 and excel_files[0][0] == "banks_components_analysis.xlsx":
    print("[OK] Only banks_components_analysis.xlsx Excel file - PERFECT!")
    if len(csv_files) <= 1 and (not csv_files or csv_files[0][0] == "all_banks_summary.csv"):
        print("[OK] CSV files are minimal (only all_banks_summary.csv preserved)")
    else:
        print("[WARN] Extra CSV files detected - they should be removed")
else:
    print("[ERROR] Unexpected Excel files in outputs")
print("="*80)
