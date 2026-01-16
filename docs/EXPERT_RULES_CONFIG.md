# Expert Rules Configuration

This project supports configurable expert rules and thresholds (e.g., Basel III CAR, LCR, NPL). Rules are externalized to a JSON file so you can update them without code changes.

## Location

- Config file: `expert_rules.json`
- Loader module: `expert_rules_config.py`
- Optional env var: `EXPERT_RULES_PATH` (override path to JSON)

## Default Schema

```json
{
  "capital_adequacy": {
    "min_car": 0.105,
    "target_car": 0.125,
    "severity": "CRITICAL"
  },
  "asset_quality": {
    "max_npl_ratio": 0.03,
    "min_coverage_ratio": 0.70,
    "severity": "HIGH"
  },
  "liquidity": {
    "min_lcr": 1.0,
    "liquidity_ratio_min": 0.30,
    "survival_days_min": 30,
    "max_loan_to_deposit": 0.95,
    "severity": "HIGH"
  },
  "profitability": {
    "min_roa": 0.005,
    "min_roe": 0.08,
    "severity": "MEDIUM"
  },
  "growth_limits": {
    "max_loan_growth": 0.30,
    "max_asset_growth": 0.25,
    "severity": "MEDIUM"
  }
}
```

## Usage

### Load Rules
```python
from expert_rules_config import load_expert_rules, get_rule
rules = load_expert_rules()  # loads expert_rules.json or defaults
min_car = get_rule(rules, 'capital_adequacy', 'min_car', 0.105)
```

### Override Path
Set `EXPERT_RULES_PATH` to point to a different JSON (e.g., jurisdiction-specific):

```powershell
$env:EXPERT_RULES_PATH = "D:\\configs\\expert_rules_vn.json"
```

### Integration Points
- `5_bank_audit_system.py`: `get_expert_rules()` now loads from the config
- `2_model_credit_risk.py`: NPL, coverage, CAR, LTD thresholds use config values
- `2_model_liquidity_risk.py`: Liquidity ratio, LTD, stress pass criteria use config values

## Examples

- Change NPL tolerance to 4%:
```json
{
  "asset_quality": { "max_npl_ratio": 0.04 }
}
```
- Increase CAR min to 12% for D-SIBs:
```json
{
  "capital_adequacy": { "min_car": 0.12 }
}
```
- Raise LCR to 110%:
```json
{
  "liquidity": { "min_lcr": 1.1 }
}
```

## Validation
The loader performs a deep merge of your JSON onto defaults. Missing keys are filled in automatically.

## Notes
- JSON is preferred for simplicity; YAML can be added if needed.
- Keep thresholds consistent across indicators and stress tests.

## Troubleshooting
- If the config fails to load, defaults are used.
- Verify your JSON syntax; use an online validator if unsure.
