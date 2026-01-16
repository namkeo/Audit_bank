# Regulatory Transparency Features - Quick Reference

## Overview
All ML models now provide **explainable outputs** with regulatory narratives that tie predictions back to specific financial metrics and peer comparisons.

## Key Features

### 1. Peer Benchmarks
Every model tracks industry-wide statistics:
- **Mean**: Industry average
- **Median**: 50th percentile  
- **Std**: Standard deviation
- **P25/P75**: 25th and 75th percentiles
- **Z-Score**: Statistical significance measure

### 2. Regulatory Narratives
Human-readable explanations in every risk assessment:
```python
"Credit Risk Assessment: HIGH (score: 72.50). Models flagging elevated 
risk: random_forest, xgboost. Key concerns: NPL ratio: 0.0650 vs peer 
average 0.0280 (above by 132% - NPL ratio above peer 75th percentile)."
```

### 3. Detailed Explanations
Structured data for programmatic access:
```python
{
  'key_factors': [
    {
      'metric': 'npl_ratio',
      'value': 0.0650,
      'peer_mean': 0.0280,
      'z_score': 4.51,
      'pct_deviation': 132.0,
      'reason': 'NPL ratio above peer 75th percentile'
    }
  ],
  'narrative': '...',
  'model_insights': {...}
}
```

## How to Access Explanations

### In Code
```python
# Run audit
report = audit_system.run_complete_audit(df, 'VCB', df)

# Credit risk explanation
credit_narrative = report['risk_assessments']['credit_risk']['regulatory_narrative']
credit_details = report['risk_assessments']['credit_risk']['ml_explanation']

# Liquidity risk explanation  
liquidity_narrative = report['risk_assessments']['liquidity_risk']['regulatory_narrative']
liquidity_details = report['risk_assessments']['liquidity_risk']['ml_explanation']

# Anomaly explanations
anomalies = report['risk_assessments']['anomaly_detection']['explained_anomalies']
for anom in anomalies:
    print(anom['regulatory_narrative'])
    print(anom['top_factors'])
```

### In Reports
All comprehensive reports automatically include:
- `regulatory_narrative`: Text explanation
- `ml_explanation`: Detailed factors and metrics
- `explained_anomalies`: Anomaly-specific explanations

## Explanation Components

### Credit Risk
- **Model Insights**: Which ML models flagged risk
- **Key Factors**: Top 5 problematic metrics with peer comparisons
- **Regulatory Context**: Compliance thresholds (NPL > 5%, CAR < 8%)

### Liquidity Risk  
- **Stress Indicators**: Which scenarios failed (baseline/stress/severe)
- **ML Stress Detection**: Anomaly-based stress patterns
- **Key Factors**: LCR, NSFR, liquidity ratio deviations
- **Basel III Context**: Regulatory requirements

### Anomaly Detection
- **Model Consensus**: Which models agreed (isolation_forest, dbscan, etc.)
- **Top Contributors**: Up to 5 features with highest z-scores
- **Deviation Details**: Percentage and statistical deviation from peers

## Interpreting Z-Scores

| Z-Score | Interpretation |
|---------|----------------|
| < ±2    | Normal range |
| ±2 to ±3 | Notable deviation |
| > ±3    | Extreme deviation |

**System flags deviations > ±2 for investigation**

## Example Use Cases

### 1. Regulatory Submission
```python
# Get compliance-ready explanation
narrative = report['risk_assessments']['credit_risk']['regulatory_narrative']

# Include in regulatory report
regulatory_doc.add_section("ML Risk Assessment", narrative)
```

### 2. Management Review
```python
# Identify urgent issues
for factor in credit_details['key_factors']:
    if abs(factor['z_score']) > 3:
        print(f"URGENT: {factor['metric']} - {factor['reason']}")
```

### 3. Audit Trail
```python
# Save detailed explanation for audit
import json
with open('audit_trail.json', 'w') as f:
    json.dump(report['risk_assessments'], f, indent=2)
```

## Benefits

✅ **Regulatory Compliance**: Every ML decision is explainable  
✅ **Transparency**: Clear connection to financial metrics  
✅ **Actionability**: Specific areas identified for improvement  
✅ **Defensibility**: Statistical rigor with peer comparisons  
✅ **Auditability**: Complete trace from prediction to source data  

## FAQs

**Q: Do I need to change existing code?**  
A: No, all changes are additive. New fields are available but optional.

**Q: What if no anomalies are detected?**  
A: System correctly reports "No anomalies detected" with no explanations needed.

**Q: Are explanations available for batch processing?**  
A: Yes, every audit includes explanations regardless of processing mode.

**Q: Can I customize the narrative format?**  
A: Yes, modify `_generate_*_explanation()` methods in risk model files.

**Q: What if peer benchmarks aren't available?**  
A: System gracefully handles missing benchmarks with fallback to absolute thresholds.

## Related Documentation

- [REGULATORY_TRANSPARENCY_COMPLETION_REPORT.md](REGULATORY_TRANSPARENCY_COMPLETION_REPORT.md) - Full implementation details
- [00_START_HERE_REPRODUCIBILITY.md](00_START_HERE_REPRODUCIBILITY.md) - System overview
- [PROJECT_INDEX.md](PROJECT_INDEX.md) - Complete documentation index

## Testing

Run transparency test:
```bash
python test_regulatory_transparency.py
```

Expected output:
```
✓ Credit risk explanation exists
✓ Liquidity risk explanation exists
✓ Anomaly explanations structure included
```
