# New Risk Indicators - Implementation Guide

## Overview

Added three categories of advanced risk indicators to enhance the bank audit system's risk detection capabilities:

1. **Net Interest Margin (NIM)** - Profitability indicator
2. **Off-Balance Sheet (OBS) Exposures** - Hidden risk indicator  
3. **Concentration Metrics** - Diversification risk indicators

---

## 1. Net Interest Margin (NIM)

### Definition
Net Interest Margin measures the profitability of a bank's lending activities:

```
NIM = (Interest Income - Interest Expense) / Average Earning Assets
```

### Why It Matters
- **Profitability Indicator**: Core measure of banking profitability
- **Interest Rate Risk**: Shows sensitivity to rate changes
- **Competitive Position**: Banks with low NIM may be under pressure
- **Regulatory Signal**: Persistently low NIM (<1.5%) indicates stress

### Thresholds
- **Minimum**: 1.5% (regulatory threshold in expert_rules.json)
- **Healthy Range**: 2.5% - 4.5%
- **Warning**: <2.0%

### New Dataset Columns
| Column | Description |
|--------|-------------|
| `interest_income` | Total interest earned on loans and investments |
| `interest_expense` | Interest paid on deposits and borrowings |
| `net_interest_margin` | (Interest Income - Interest Expense) / Total Assets |

### Implementation
- **Added to**: `4_utility_functions.py` - `FinancialRatioCalculator.calculate_ratios()`
- **Expert Rules**: `expert_rules.json` - `profitability.min_net_interest_margin`
- **Risk Models**: Used in credit risk and profitability assessments

---

## 2. Off-Balance Sheet (OBS) Exposures

### Definition
Off-balance sheet items are financial obligations not reflected on the balance sheet but create contingent liabilities:

- **Derivatives**: Swaps, options, futures (notional value)
- **Unused Credit Lines**: Committed but undrawn credit facilities
- **Guarantees**: Letters of credit, performance bonds

### Why It Matters
- **Hidden Liquidity Risk**: Can be drawn suddenly in crisis
- **Counterparty Risk**: Derivatives expose banks to trading partner defaults
- **Leverage Amplification**: Large OBS positions magnify balance sheet risk
- **Regulatory Capital**: Basel III requires capital against OBS exposures

### Real-World Examples
- **2008 Financial Crisis**: Banks had massive OBS derivatives (>10x assets)
- **Silicon Valley Bank (2023)**: Large unrealized losses in OBS securities portfolio

### Thresholds
| Metric | Safe | Warning | Critical |
|--------|------|---------|----------|
| Derivatives/Assets | <0.5 | 0.5-1.5 | >2.0 |
| Unused Lines/Loans | <0.5 | 0.5-0.8 | >0.8 |
| Total OBS/Assets | <0.5 | 0.5-1.5 | >1.5 |

### New Dataset Columns
| Column | Description | Risk Implication |
|--------|-------------|------------------|
| `derivatives_notional` | Total notional value of derivatives | Counterparty & market risk |
| `unused_credit_lines` | Committed but undrawn credit | Liquidity call risk |
| `guarantees_issued` | Letters of credit, guarantees | Contingent liability |
| `derivatives_to_assets_ratio` | Derivatives / Total Assets | Relative exposure |
| `unused_lines_to_loans_ratio` | Unused Lines / Total Loans | Commitment risk |
| `obs_to_assets_ratio` | Total OBS / Total Assets | Overall OBS leverage |
| `obs_risk_indicator` | Composite 0-1 score | Aggregate OBS risk |

### Calculation Formula
```python
# Credit Equivalent Amount (Basel III conversion factors)
obs_exposure_total = (
    derivatives_notional * 0.05 +    # 5% conversion (credit equivalent)
    unused_credit_lines * 0.5 +      # 50% conversion (likely drawdown)
    guarantees_issued * 0.8          # 80% conversion (high probability)
)
```

### Implementation
- **Added to**: `4_utility_functions.py` - OBS ratios in `calculate_ratios()`
- **Expert Rules**: `expert_rules.json` - `off_balance_sheet` section
- **Risk Models**: Liquidity risk model considers OBS drawdown scenarios

---

## 3. Concentration Metrics

### Definition
Concentration risk arises when a bank's exposures are not sufficiently diversified:

### 3.1 Deposit Concentration

**Top 20 Depositors Ratio**:
```
Top 20 Depositors / Total Deposits
```

**Why It Matters**:
- **Liquidity Risk**: If 50%+ deposits from 20 clients, withdrawal risk is high
- **Funding Stability**: Concentrated deposit base = volatile funding
- **Regulatory Concern**: Banks must maintain diverse deposit base

**Thresholds**:
- Safe: <30%
- Warning: 30-50%
- Critical: >50%

### 3.2 Loan Concentration

#### a) Sector Concentration (HHI)

**Herfindahl-Hirschman Index**:
```
HHI = Σ (sector_share²)
```

**Interpretation**:
- **0.0 - 0.15**: Highly diversified
- **0.15 - 0.25**: Moderate concentration
- **0.25 - 0.35**: High concentration (warning)
- **>0.35**: Very high concentration (critical)

**Why It Matters**:
- **Sectoral Shocks**: If real estate = 60% of loans and housing crashes, bank fails
- **Systemic Risk**: Concentrated lending amplifies economic cycles
- **Capital Requirements**: Regulators require extra capital for concentrated portfolios

#### b) Borrower Concentration

**Top 20 Borrowers / Total Loans**

**Thresholds**:
- Safe: <15%
- Warning: 15-25%
- Critical: >25%

**Why It Matters**:
- **Single Point of Failure**: One borrower default = major loss
- **Regulatory Limits**: Many jurisdictions limit single borrower exposure to 10-15% of capital

#### c) Geographic Concentration

**Scale**: 0 (diversified) to 1.0 (single region)

**Why It Matters**:
- **Regional Shocks**: Natural disasters, local economic downturns
- **Diversification Benefit**: Multi-region banks weather local crises better

### New Dataset Columns

| Column | Description | Threshold |
|--------|-------------|-----------|
| `top20_depositors` | Total deposits from top 20 clients | - |
| `top20_depositors_ratio` | Top 20 / Total Deposits | <0.50 |
| `top5_depositors` | Top 5 depositor concentration | - |
| `top5_depositors_ratio` | Top 5 / Total Deposits | <0.30 |
| `sector_concentration_hhi` | HHI for loan sectors | <0.35 |
| `top20_borrower_loans` | Loans to top 20 borrowers | - |
| `top20_borrower_concentration` | Top 20 Borrowers / Loans | <0.25 |
| `geographic_concentration` | Regional concentration (0-1) | <0.75 |

### Implementation
- **Added to**: `4_utility_functions.py` - Concentration ratios
- **Expert Rules**: `expert_rules.json` - `concentration_limits` section
- **Risk Models**: Credit and liquidity models use in risk scoring

---

## 4. Composite Risk Indicators

### 4.1 Liquidity-Concentration Risk
```python
liquidity_concentration_risk = (
    top20_depositors_ratio * 0.4 +         # Deposit concentration
    wholesale_dependency_ratio * 0.3 +     # Wholesale funding risk
    (1 - min(LCR, 2) / 2) * 0.3           # LCR deficiency
)
```

**Interpretation**: 0-1 score, >0.60 is high risk

### 4.2 Credit Concentration Risk
```python
credit_concentration_risk = (
    sector_concentration_hhi * 0.5 +      # Sector concentration
    top20_borrower_concentration * 0.3 +  # Borrower concentration
    geographic_concentration * 0.2        # Geographic concentration
)
```

**Interpretation**: 0-1 score, >0.50 is high risk

### 4.3 OBS Risk Indicator
```python
obs_risk_indicator = (
    min(derivatives_to_assets, 3) / 3 * 0.5 +  # Derivatives risk
    min(unused_lines_to_loans, 1) * 0.3 +      # Commitment risk
    min(obs_to_assets, 2) / 2 * 0.2           # Total OBS risk
)
```

**Interpretation**: 0-1 score, >0.70 is high risk

---

## Usage Examples

### 1. Check NIM for a Bank
```python
from 4_utility_functions import FinancialRatioCalculator

ratios = FinancialRatioCalculator.calculate_ratios(bank_data)

if ratios['net_interest_margin'] < 0.015:
    print("WARNING: NIM below regulatory minimum")
```

### 2. Assess OBS Risk
```python
if ratios['derivatives_to_assets_ratio'] > 2.0:
    print("CRITICAL: Excessive derivatives exposure")

if ratios['obs_risk_indicator'] > 0.7:
    print("HIGH: Composite OBS risk elevated")
```

### 3. Evaluate Concentration Risk
```python
# Deposit concentration
if ratios['top20_depositors_ratio'] > 0.5:
    print("HIGH RISK: Deposits concentrated in top 20 clients")

# Loan concentration
if ratios['sector_concentration_hhi'] > 0.35:
    print("CRITICAL: Loan portfolio lacks diversification")

if ratios['top20_borrower_concentration'] > 0.25:
    print("WARNING: High borrower concentration")
```

### 4. Full Risk Assessment
```python
# Run audit with new indicators
audit_system = BankAuditSystem(bank_name='VCB', audit_period='2024-Q1')
report = audit_system.run_complete_audit(df, 'VCB', df)

# Access new indicators in report
credit_risk = report['risk_assessments']['credit_risk']
print(f"Credit Concentration Risk: {credit_risk['indicators'].get('credit_concentration_risk', 'N/A')}")

liquidity_risk = report['risk_assessments']['liquidity_risk']
print(f"Liquidity Concentration Risk: {liquidity_risk['indicators'].get('liquidity_concentration_risk', 'N/A')}")
```

---

## Regulatory Context

### Basel III Requirements

1. **OBS Exposures**: Must be converted to credit equivalents and included in risk-weighted assets
2. **Concentration Limits**: Large exposure framework limits single borrower to 25% of Tier 1 capital
3. **NSFR**: Includes OBS commitments in required stable funding calculation

### Banking Regulations

- **Federal Reserve**: Requires disclosure of top 20 depositor concentrations
- **OCC**: Examines loan concentration as part of credit risk assessment
- **FDIC**: Considers deposit concentration in insurance premium calculations

---

## Files Modified

1. **add_new_indicators.py** (NEW): Script to enrich dataset
2. **time_series_dataset_enriched_v2.csv** (NEW): Enhanced dataset (66 columns, was 45)
3. **4_utility_functions.py**: Added new ratios to `calculate_ratios()`
4. **expert_rules.json**: Added thresholds for new indicators
5. **0_example_usage.py**: Updated to use new dataset

---

## Testing

Run the enrichment script:
```bash
python add_new_indicators.py
```

Expected output:
```
✓ Added: interest_income, interest_expense, net_interest_margin
✓ Added: derivatives_notional, unused_credit_lines, guarantees_issued
✓ Added: top20_depositors, top5_depositors with ratios
✓ Added: sector_concentration_hhi
✓ Enhancement Complete!
```

Verify in example:
```bash
python 0_example_usage.py
```

---

## Summary

| Category | Indicators Added | Key Threshold |
|----------|------------------|---------------|
| **NIM** | 3 | Min: 1.5% |
| **OBS** | 6 | Derivatives/Assets <2.0 |
| **Concentration** | 8 | Top 20 Depositors <50% |
| **Composite** | 3 | Risk scores <0.6-0.7 |
| **Total** | **21 new indicators** | - |

Dataset expanded from **45 to 66 columns** (+47% enrichment)

All indicators integrated into:
- ✅ Ratio calculation system
- ✅ Expert rule thresholds  
- ✅ Risk model assessments
- ✅ Regulatory explanations
