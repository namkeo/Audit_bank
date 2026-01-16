# Bank Audit System

## Overview

A comprehensive ML-powered financial risk assessment platform for regulatory compliance and internal bank audits. Features ensemble anomaly detection, credit risk modeling, liquidity risk assessment, and full regulatory transparency with peer clustering.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run example audit
python 0_example_usage.py
```

## Project Structure

```
bank-audit-system/
├── config/          # Configuration & expert rules
├── data/            # Datasets
├── docs/            # Documentation (33 guides)
├── logs/            # Application logs
├── outputs/         # Generated reports & dashboards
├── tests/           # Test suite (11 tests)
├── utils/           # Utilities (batch, enrichment, etc.)
├── 0_example_usage.py           # Examples & demos
├── 1_data_preparation.py        # Data pipeline
├── 2_model_*.py                 # ML models (anomaly, credit, liquidity)
├── 3_reporting_analysis.py      # Report generation
├── 4_utility_functions.py       # Shared utilities
├── 5_bank_audit_system.py       # Main orchestrator
├── 6_logging_config.py          # Logging configuration
├── ARCHITECTURE.md              # Full architecture documentation
└── requirements.txt             # Dependencies
```

## Key Features

- **Ensemble Anomaly Detection**: 5 ML algorithms with voting mechanism
- **Peer Clustering**: KMeans-based grouping with dynamic k selection
- **Regulatory Transparency**: Full explainability & peer benchmarks
- **Multi-Risk Assessment**: Credit, liquidity, operational risks
- **Batch Processing**: Vectorized operations for large-scale audits
- **66+ Financial Ratios**: Including NIM, OBS, concentration metrics
- **Comprehensive Testing**: 11 test modules, 33 passing tests

## Core Components

| Component | Purpose | Key Algorithms |
|-----------|---------|----------------|
| **Anomaly Detection** | Detect unusual patterns | IsolationForest, DBSCAN, SVM, LOF, Elliptic Envelope |
| **Credit Risk** | Assess credit quality | RandomForest, GradientBoosting, XGBoost |
| **Liquidity Risk** | Evaluate liquidity position | Stress testing, LCR, NSFR, forecasting |
| **Peer Clustering** | Group similar banks | KMeans with silhouette/CH optimization |
| **Reporting** | Generate audit reports | JSON, Excel, PDF, HTML, dashboards |

## Example Usage

```python
from bank_audit_system import BankAuditSystem
import pandas as pd

# Load data
df = pd.read_csv('data/time_series_dataset_enriched_v2.csv')

# Run audit
audit = BankAuditSystem(bank_name="VCB", audit_period="2024-Q1")
report = audit.run_complete_audit(df=df, bank_id="VCB", all_banks_data=df)

# Export results
audit.export_results("outputs/report.json", format='json')
audit.create_dashboard(report, save_path="outputs/dashboard.png")
```

## Documentation

- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for complete system design
- **Getting Started**: See [docs/00_START_HERE_REPRODUCIBILITY.md](docs/00_START_HERE_REPRODUCIBILITY.md)
- **Regulatory Transparency**: [docs/REGULATORY_TRANSPARENCY_SUMMARY.md](docs/REGULATORY_TRANSPARENCY_SUMMARY.md)
- **New Indicators**: [docs/NEW_INDICATORS_GUIDE.md](docs/NEW_INDICATORS_GUIDE.md)
- **Batch Processing**: [docs/BATCH_PROCESSING_GUIDE.md](docs/BATCH_PROCESSING_GUIDE.md)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_regulatory_transparency.py -v
```

## Technology Stack

- Python 3.8+
- Scikit-learn (ML models, clustering, metrics)
- Pandas, NumPy (data processing)
- XGBoost (gradient boosting)
- Matplotlib, Seaborn (visualization)
- Pytest (testing)

## Recent Enhancements

1. **Peer Clustering** (Jan 2026): KMeans-based grouping with dynamic k selection
2. **Regulatory Transparency** (Dec 2025): ML explanations & narratives
3. **Dataset Enrichment** (Dec 2025): NIM, OBS, concentration metrics
4. **Batch Processing** (Nov 2025): Vectorized operations (10-100x faster)

## License

[Specify License]

---

**Version**: 1.1.0  
**Last Updated**: January 10, 2026
