#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Show banks with expert rule violations
"""
import json
import os
import pandas as pd
from pathlib import Path

os.chdir(r"D:\2026\KTNN\Research audit bank\new source vscode 1.1")

print("="*80)
print("EXPERT RULES VIOLATIONS ANALYSIS - ALL BANKS")
print("="*80)

# Load expert rules for reference
with open('config/expert_rules.json', 'r') as f:
    expert_rules = json.load(f)

print("\nEXPERT RULES THRESHOLDS:")
print("-"*80)
print(f"Capital Adequacy (CAR):     min {expert_rules['capital_adequacy']['min_car']:.2%}")
print(f"Asset Quality (NPL):        max {expert_rules['asset_quality']['max_npl_ratio']:.2%}")
print(f"Liquidity (LTD):            max {expert_rules['liquidity']['max_loan_to_deposit']:.2%}")
print(f"Profitability (ROA):        min {expert_rules['profitability']['min_roa']:.2%}")
print()

# Collect all violations by bank
all_violations = {}
bank_ids = []

# Load bank reports
for file in sorted(os.listdir("outputs")):
    if file.endswith("_bank_report.json"):
        bank_id = file.replace("_bank_report.json", "").upper()
        bank_ids.append(bank_id)
        
        with open(f"outputs/{file}", 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        violations = report.get('expert_rules_violations', [])
        if violations:
            all_violations[bank_id] = violations

# Display results
print("\n" + "="*80)
print("BANKS WITH VIOLATIONS")
print("="*80)

if not all_violations:
    print("\nGood News! No banks are currently violating expert rules.")
    print("\nAll 10 banks are in compliance with regulatory thresholds:")
    for bank_id in sorted(bank_ids):
        print(f"  ✓ {bank_id}")
else:
    print(f"\nFound {len(all_violations)} bank(s) with violations:\n")
    
    for bank_id in sorted(all_violations.keys()):
        violations = all_violations[bank_id]
        print(f"\n{'='*80}")
        print(f"BANK: {bank_id}")
        print(f"Total Violations: {len(violations)}")
        print(f"{'='*80}")
        
        for idx, violation in enumerate(violations, 1):
            print(f"\n  Violation {idx}:")
            print(f"    Rule:       {violation.get('rule', 'N/A')}")
            print(f"    Metric:     {violation.get('metric', 'N/A')}")
            print(f"    Value:      {violation.get('value', 'N/A'):.4f}")
            print(f"    Threshold:  {violation.get('threshold', 'N/A'):.4f}")
            print(f"    Severity:   {violation.get('severity', 'N/A')}")
            print(f"    Message:    {violation.get('message', 'N/A')}")

# Create summary table
print("\n" + "="*80)
print("VIOLATION SUMMARY TABLE")
print("="*80)

summary_data = []
for bank_id in sorted(bank_ids):
    violations = all_violations.get(bank_id, [])
    
    violation_types = {}
    for v in violations:
        rule = v.get('rule', 'Unknown')
        violation_types[rule] = violation_types.get(rule, 0) + 1
    
    summary_data.append({
        'Bank ID': bank_id,
        'Total Violations': len(violations),
        'Capital Adequacy': violation_types.get('Capital Adequacy', 0),
        'Asset Quality': violation_types.get('Asset Quality', 0),
        'Liquidity': violation_types.get('Liquidity', 0),
        'Profitability': violation_types.get('Profitability', 0),
        'Status': 'COMPLIANT' if len(violations) == 0 else 'VIOLATIONS'
    })

summary_df = pd.DataFrame(summary_data)
print("\n")
print(summary_df.to_string(index=False))

# Export to Excel if all banks are compliant
print("\n" + "="*80)
print("EXPORT OPTIONS")
print("="*80)

violations_by_type = {}
for bank_id, violations in all_violations.items():
    for v in violations:
        rule = v.get('rule', 'Unknown')
        if rule not in violations_by_type:
            violations_by_type[rule] = []
        violations_by_type[rule].append({
            'Bank ID': bank_id,
            'Metric': v.get('metric'),
            'Value': v.get('value'),
            'Threshold': v.get('threshold'),
            'Severity': v.get('severity'),
            'Message': v.get('message')
        })

if violations_by_type:
    # Create Excel with violations
    excel_path = "outputs/expert_rules_violations.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Violations by type
        for rule_type, violations_list in sorted(violations_by_type.items()):
            df = pd.DataFrame(violations_list)
            sheet_name = rule_type.replace(' ', '_')[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n✓ Violations exported to: {excel_path}")
else:
    print("\n✓ All banks are compliant - no violations to export")

print("\n" + "="*80)
