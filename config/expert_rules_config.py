import os
import json
from typing import Dict, Optional

DEFAULT_RULES: Dict = {
    "capital_adequacy": {
        "min_car": 0.105,
        "target_car": 0.125,
        "severity": "CRITICAL",
    },
    "asset_quality": {
        "max_npl_ratio": 0.03,
        "min_coverage_ratio": 0.70,
        "severity": "HIGH",
    },
    "liquidity": {
        "min_lcr": 1.0,
        "liquidity_ratio_min": 0.30,
        "survival_days_min": 30,
        "max_loan_to_deposit": 0.85,
        "severity": "HIGH",
    },
    "profitability": {
        "min_roa": 0.005,
        "min_roe": 0.08,
        "severity": "MEDIUM",
    },
    "growth_limits": {
        "max_loan_growth": 0.30,
        "max_asset_growth": 0.25,
        "severity": "MEDIUM",
    },
    "concentration": {
        "max_sector_hhi": 0.20,
        "max_top_borrower_ratio": 0.10,
        "max_top10_borrower_ratio": 0.35,
        "severity": "HIGH"
    },
    "off_balance": {
        "max_obs_to_loans_ratio": 0.25,
        "severity": "MEDIUM"
    },
    "external": {
        "min_rating_notch": 5,
        "severity": "INFO"
    }
}

CONFIG_ENV_VAR = "EXPERT_RULES_PATH"
# Since this file is now in config/, expert_rules.json is in the same directory
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "expert_rules.json")


def _deep_update(base: Dict, overrides: Dict) -> Dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_expert_rules(path: Optional[str] = None) -> Dict:
    """
    Tải các quy tắc chuyên gia từ tệp JSON, quay lại DEFAULT_RULES.
    Đường dẫn ghi đè tùy chọn qua ENV `EXPERT_RULES_PATH`.
    
    Args:
        path (Optional[str]): Đường dẫn tùy chọn tới tệp cấu hình quy tắc
        
    Returns:
        dict: Từ điển các quy tắc chuyên gia với giá trị mặc định hoặc từ tệp
        
    Raises:
        JSONDecodeError: Nếu tệp cấu hình không hợp lệ
    """
    config_path = path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    rules = json.loads(json.dumps(DEFAULT_RULES))  # deep copy via JSON
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = _deep_update(rules, data)
    except Exception:
        # On any error, return defaults
        return rules
    return rules


def get_rule(rules: Dict, category: str, key: str, default=None):
    return rules.get(category, {}).get(key, default)


def print_rules_summary(rules: Dict) -> None:
    print("Expert Rules Summary:")
    for cat, vals in rules.items():
        keys = ", ".join(sorted(vals.keys()))
        print(f" - {cat}: {keys}")
