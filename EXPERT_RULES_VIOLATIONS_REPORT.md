# Expert Rules Violations Report

## Summary

**All 10 banks currently analyzed have violations of expert rules**, specifically related to **Capital Adequacy Ratio (CAR)**.

---

## Violation Details

### Critical Issue: Capital Adequacy Ratio (CAR) Below Minimum

**Regulatory Requirement:** CAR must be at least **10.50%**

All 10 banks are significantly below this threshold, with values ranging from **3.39% to 4.63%**:

| Bank ID | CAR Value | Requirement | Shortfall | Severity |
|---------|-----------|-------------|-----------|----------|
| HDB     | 3.51%     | 10.50%      | -6.99%    | **CRITICAL** |
| CTG     | 3.39%     | 10.50%      | -7.11%    | **CRITICAL** |
| VCB     | 3.74%     | 10.50%      | -6.76%    | **CRITICAL** |
| EIB     | 4.31%     | 10.50%      | -6.19%    | **CRITICAL** |
| MBB     | 4.13%     | 10.50%      | -6.37%    | **CRITICAL** |
| BIDV    | 4.20%     | 10.50%      | -6.30%    | **CRITICAL** |
| ACB     | 4.63%     | 10.50%      | -5.87%    | **CRITICAL** |
| STB     | 4.27%     | 10.50%      | -6.23%    | **CRITICAL** |
| TCB     | 4.51%     | 10.50%      | -5.99%    | **CRITICAL** |
| VPB     | 4.59%     | 10.50%      | -5.91%    | **CRITICAL** |

---

## Compliance Status

| Category | Count | Compliance |
|----------|-------|-----------|
| Banks Compliant | 0 | ❌ |
| Banks Violating | 10 | ⚠️ |
| **Total Banks** | **10** | **0% Compliant** |

---

## Expert Rules Configuration

The following rules are being monitored:

### 1. **Capital Adequacy** (CRITICAL)
   - **Rule:** CAR (Capital Adequacy Ratio) >= 10.50%
   - **Status:** ❌ ALL 10 BANKS VIOLATING
   - **Average Bank CAR:** 4.13%
   - **Shortfall:** -155% below requirement

### 2. **Asset Quality** (HIGH)
   - **Rule:** NPL Ratio <= 3.00%
   - **Status:** ✓ COMPLIANT

### 3. **Liquidity** (HIGH)
   - **Rule:** Loan-to-Deposit Ratio <= 85.00%
   - **Status:** ✓ COMPLIANT

### 4. **Profitability** (MEDIUM)
   - **Rule:** ROA (Return on Assets) >= 0.50%
   - **Status:** ✓ COMPLIANT

---

## Recommendations

### Immediate Action Required (Critical Priority)

1. **Capital Adequacy Improvement**
   - All banks need to increase their CAR from ~4% to 10.5% minimum
   - This requires either:
     - **Increase capital:** Raise more equity capital
     - **Reduce risk-weighted assets:** Reduce high-risk lending or improve loan quality
     - **Reduce leverage:** Lower loan-to-deposit ratios and reduce asset growth

2. **Regulatory Notification**
   - Notify the banking regulator of systemic capital adequacy issues
   - This may trigger regulatory intervention or restructuring requirements

3. **Capital Injection**
   - All 10 banks should seek capital injection or equity investment
   - Average shortfall is ~6-7% per bank across the system

### Bank-Specific Concerns

**Highest Priority (Lowest CAR):**
- 🔴 **HDB** (3.51%) - Largest shortfall
- 🔴 **CTG** (3.39%) - Second-lowest CAR
- 🔴 **VCB** (3.74%)

**Lower Priority (Higher CAR but still violating):**
- 🟡 **ACB** (4.63%)
- 🟡 **VPB** (4.59%)
- 🟡 **TCB** (4.51%)

---

## Output Files

| File | Location | Contents |
|------|----------|----------|
| **expert_rules_violations.xlsx** | outputs/ | Detailed violations by rule type |
| **Bank Reports** | outputs/*_bank_report.json | Individual bank violation details |

---

## Next Steps

1. ✅ Run `show_violations.py` to generate this report
2. ✅ Review `outputs/expert_rules_violations.xlsx` for detailed breakdown
3. ✅ Check individual bank reports for full context
4. 📋 **TODO:** Develop capital improvement plans for each bank
5. 📋 **TODO:** Notify regulatory authorities
6. 📋 **TODO:** Implement corrective action plans

---

## Technical Details

**Rule Engine:** Expert Rules System  
**Rules Configuration:** `config/expert_rules.json`  
**Validation Method:** Applied during comprehensive audit  
**Detection Timing:** Latest financial period data  
**Severity Classification:** CRITICAL / HIGH / MEDIUM / LOW

---

Generated: 2026-01-16  
Analysis Tool: Bank Audit System v1.1
