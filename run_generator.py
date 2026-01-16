#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick script to verify the outputs folder contains only banks_components_analysis.xlsx
"""
import os
import sys

os.chdir(r"D:\2026\KTNN\Research audit bank\new source vscode 1.1")

print("="*80)
print("OUTPUTS FOLDER CONTENTS:")
print("="*80)

excel_files = []
csv_files = []
other_files = []

for fname in sorted(os.listdir("outputs")):
    fpath = os.path.join("outputs", fname)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        if fname.lower().endswith(".xlsx"):
            excel_files.append((fname, size))
        elif fname.lower().endswith(".csv"):
            csv_files.append((fname, size))
        else:
            other_files.append((fname, size))

print("\nExcel Files:")
if excel_files:
    for fname, size in excel_files:
        print(f"  [EXCEL] {fname:<40} ({size:>10,} bytes)")
else:
    print("  (none)")

print("\nCSV Files (should be removed):")
if csv_files:
    print("  [WARNING] The following CSV files should have been removed:")
    for fname, size in csv_files:
        print(f"  [CSV]    {fname:<40} ({size:>10,} bytes)")
else:
    print("  (none - GOOD!)")

print("\nOther Files:")
if other_files:
    for fname, size in other_files[:5]:  # Show first 5
        print(f"  {fname:<40} ({size:>10,} bytes)")
    if len(other_files) > 5:
        print(f"  ... and {len(other_files) - 5} more files")
else:
    print("  (none)")

print("\n" + "="*80)
if len(excel_files) == 1 and excel_files[0][0] == "banks_components_analysis.xlsx":
    print("SUCCESS: Only banks_components_analysis.xlsx exists!")
    if csv_files:
        print("NOTE: Old CSV files detected - they should be removed on next run")
else:
    print("WARNING: Output files need adjustment")
print("="*80)

