# Where is the Overall Risk Score Calculated?

## Quick Answer

**The overall risk score is calculated using TIME-SERIES DATA** - it processes all historical data points (20 periods) for each bank and generates a composite score based on:

1. **Credit Risk Score** (40% weight) - calculated from all 20 time periods
2. **Liquidity Risk Score** (35% weight) - calculated from all 20 time periods  
3. **Anomaly Detection Score** (25% weight) - calculated from all 20 time periods

---

## Complete Calculation Flow

### Step 1: Data Loading
**File:** `5_bank_audit_system.py` → `load_and_prepare_data()` method

```python
# Load TIME-SERIES data for the bank (20 periods)
time_series_data = self.data_prep.load_time_series_data(df, bank_id, time_column='period')

# Calculate RATIOS from all 20 periods
ratios_df = self.data_prep.calculate_time_series_ratios()

# Prepare FEATURES from all 20 periods
features_df = self.data_prep.prepare_time_series_features()
```

**What this means:**
- ✅ **NOT** just the latest quarter/year
- ✅ **YES** all historical data (20 time periods)
- Data structure: Each bank has 20 rows of data representing different time periods

---

### Step 2: Model Training (Industry Baseline)
**File:** `5_bank_audit_system.py` → `train_models()` method

```python
# Train on ALL BANKS' DATA (multi-bank dataset)
all_banks_data = time_series_data  # All banks, all periods

# Train credit risk model on all banks' history
credit_training = self.credit_risk.train_models(all_banks_data)

# Train liquidity risk model on all banks' history
liquidity_training = self.liquidity_risk.train_models(all_banks_data)

# Train anomaly detection on all banks' history
anomaly_training = self.anomaly_detection.train_models(all_banks_data)
```

**What this means:**
- Models are trained on historical patterns across **all banks**
- Captures industry-wide risk patterns and benchmarks
- Each model learns from 10 banks × 20 periods = 200 data points

---

### Step 3: Individual Bank Risk Assessment
**File:** `5_bank_audit_system.py` → `assess_all_risks()` method

```python
# For EACH BANK:
# 1. Get credit risk using all 20 periods
credit_results = self.credit_risk.predict_risk(prepared_data)

# 2. Get liquidity risk using all 20 periods
liquidity_results = self.liquidity_risk.predict_risk(prepared_data)

# 3. Get anomaly detection using all 20 periods
anomaly_results = self.anomaly_detection.predict_risk(prepared_data)
```

**What this means:**
- Each risk metric uses **all 20 time periods** for the bank
- Not just the last period!
- Captures trends, deterioration, volatility

---

### Step 4: Overall Score Calculation
**File:** `3_reporting_analysis.py` → `_calculate_overall_risk_score()` method

```python
def _calculate_overall_risk_score(self, credit_results, liquidity_results, anomaly_results):
    """
    Calculate weighted composite score
    """
    
    # Extract individual component scores (each calculated from all 20 periods)
    scores = []
    weights = []
    
    # Credit Risk Component (40% weight)
    scores.append(credit_results['credit_risk_score'])
    weights.append(0.40)
    
    # Liquidity Risk Component (35% weight)
    scores.append(liquidity_results['liquidity_risk_score'])
    weights.append(0.35)
    
    # Anomaly Detection Component (25% weight)
    anomaly_score = anomaly_results['anomaly_rate'] * 100
    scores.append(anomaly_score)
    weights.append(0.25)
    
    # Calculate WEIGHTED AVERAGE
    base_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    # Apply early warning adjustments
    warning_count = count_early_warning_signals(all_results)
    if warning_count >= 3:
        warning_adjustment = 10.0  # Conservative bump
    elif warning_count >= 1:
        warning_adjustment = 5.0
    else:
        warning_adjustment = 0.0
    
    # Final overall score (0-100 scale)
    overall_score = min(100.0, base_score + warning_adjustment)
    
    return {
        'overall_score': overall_score,
        'risk_level': classify_level(overall_score),
        'component_scores': {
            'credit': credit_score,
            'liquidity': liquidity_score,
            'anomaly': anomaly_score
        }
    }
```

---

## Data Timeline Visualization

```
TIME PERIODS IN DATA:
Period 1  → Historical data point
Period 2  → Historical data point
...
Period 19 → Historical data point
Period 20 → Most recent data point
    ↓
All 20 periods feed into:
    ↓
Credit Risk Model        ↘
Liquidity Risk Model     → Overall Score Calculation
Anomaly Detection Model ↗
    ↓
Final Overall Risk Score (0-100)
```

---

## Component Score Details

### Credit Risk Score (40% weight)
**Calculation Method:** From `2_model_credit_risk.py`

Uses ALL 20 periods to evaluate:
- NPL (Non-Performing Loans) trend
- Capital Adequacy Ratio (CAR) trend
- Loan loss provisions adequacy
- Portfolio quality deterioration
- **Result:** Single score 0-100 representing overall credit risk

### Liquidity Risk Score (35% weight)
**Calculation Method:** From `2_model_liquidity_risk.py`

Uses ALL 20 periods to evaluate:
- Loan-to-Deposit Ratio trend
- Liquid Asset Coverage trend
- LCR (Liquidity Coverage Ratio) trend
- NSFR (Net Stable Funding Ratio) trend
- Wholesale dependency trend
- **Result:** Single score 0-100 representing overall liquidity risk

### Anomaly Detection Score (25% weight)
**Calculation Method:** From `2_model_anomaly_detection.py`

Uses ALL 20 periods to detect:
- Unusual patterns in financial metrics
- Outliers relative to peers
- Fraud indicators
- Data quality issues
- **Result:** Anomaly rate (0-1) converted to 0-100 scale

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Data Used** | Time-series: ALL 20 periods per bank |
| **NOT Used** | Single quarter/year only |
| **Training Data** | All 10 banks × 20 periods = 200 data points |
| **Calculation Location** | `3_reporting_analysis.py` line 77 |
| **Formula** | 40% Credit + 35% Liquidity + 25% Anomaly + Early Warnings |
| **Output Range** | 0-100 score |
| **Early Warnings** | Applied if 3+ signals detected (±5-10 points) |
| **Scale** | MINIMAL (0-30) → LOW → MEDIUM → HIGH → CRITICAL (80+) |

---

## Code References

| Component | File | Method | Lines |
|-----------|------|--------|-------|
| **Data Loading** | `5_bank_audit_system.py` | `load_and_prepare_data()` | 140-170 |
| **Model Training** | `5_bank_audit_system.py` | `train_models()` | 200-235 |
| **Risk Assessment** | `5_bank_audit_system.py` | `assess_all_risks()` | 245-280 |
| **Score Calculation** | `3_reporting_analysis.py` | `_calculate_overall_risk_score()` | 350-410 |
| **Credit Component** | `2_model_credit_risk.py` | `predict_risk()` | 330-450 |
| **Liquidity Component** | `2_model_liquidity_risk.py` | `predict_risk()` | 260-380 |
| **Anomaly Component** | `2_model_anomaly_detection.py` | `predict_risk()` | 280-400 |

---

## Example: CTG Bank Overall Score Calculation

Based on actual audit results:

```
Credit Risk Score:    11.84 (40% weight)
Liquidity Risk Score: 68.23 (35% weight)
Anomaly Score:         0.00 (25% weight)

Calculation:
= (11.84 × 0.40) + (68.23 × 0.35) + (0.00 × 0.25)
= 4.736 + 23.881 + 0.000
= 28.62 (base score)

Early Warning Signals: 0 (no adjustment)
Final Overall Score: 28.62

Risk Classification: MINIMAL
Quantile Classification: HIGH (top 25% of banks by risk)
```

---

## Answer Summary

**The overall risk score uses TIME-SERIES DATA of ALL 20 periods**, not just the most recent quarter or year. It's a composite weighted score combining:

1. **All historical time periods** for trend analysis
2. **All bank data** for peer benchmarking during training
3. **Three risk components** with specific weights
4. **Early warning signals** for emergency escalation

This approach ensures that the audit captures:
- ✅ Trends and trajectories (improving vs. deteriorating)
- ✅ Volatility and consistency issues
- ✅ Peer comparison and relative risk
- ✅ Emerging risks before they become critical

The score is NOT a snapshot but a **holistic assessment** based on historical patterns and forward-looking indicators.
