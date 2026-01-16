#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect bank report structure
"""
import json
import os

os.chdir(r"D:\2026\KTNN\Research audit bank\new source vscode 1.1")

# Load a sample bank report
with open("outputs/ctg_bank_report.json", 'r', encoding='utf-8') as f:
    report = json.load(f)

print("Bank Report Keys:")
for key in report.keys():
    print(f"  - {key}")

print("\nExpert Rules Violations:")
violations = report.get('expert_rules_violations', [])
print(f"  Count: {len(violations)}")
print(f"  Type: {type(violations)}")
print(f"  Content: {violations}")

print("\nAll bank IDs and violation counts:")
for file in sorted(os.listdir("outputs")):
    if file.endswith("_bank_report.json"):
        bank_id = file.replace("_bank_report.json", "").upper()
        with open(f"outputs/{file}", 'r', encoding='utf-8') as f:
            r = json.load(f)
        violations = r.get('expert_rules_violations', [])
        print(f"  {bank_id}: {len(violations)} violations")
