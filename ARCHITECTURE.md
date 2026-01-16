# Bank Audit System - Project Architecture

## Overview

The **Bank Audit System** is a comprehensive ML-powered financial risk assessment platform designed for regulatory compliance and internal audit purposes. The system implements multiple machine learning models for anomaly detection, credit risk assessment, and liquidity risk evaluation, with full regulatory transparency and explainability.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Key Features](#key-features)
6. [Technology Stack](#technology-stack)
7. [Getting Started](#getting-started)
8. [Usage Examples](#usage-examples)
9. [Testing](#testing)
10. [Configuration](#configuration)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bank Audit System (Orchestrator)              │
│                      5_bank_audit_system.py                      │
└────────────┬─────────────────────────────────────┬───────────────┘
             │                                     │
    ┌────────▼────────┐                   ┌────────▼────────┐
    │  Data Pipeline  │                   │  Risk Models    │
    └────────┬────────┘                   └────────┬────────┘
             │                                     │
   ┌─────────┼─────────┐             ┌────────────┼────────────┐
   │         │         │             │            │            │
┌──▼──┐  ┌──▼──┐  ┌──▼──┐      ┌───▼───┐   ┌───▼───┐   ┌───▼────┐
│Data │  │Ratio│  │Feat │      │Anomaly│   │Credit │   │Liquidity│
│Prep │  │Calc │  │Eng  │      │Detect │   │ Risk  │   │  Risk   │
└─────┘  └─────┘  └─────┘      └───────┘   └───────┘   └────────┘
                                     │           │            │
                        ┌────────────┴───────────┴────────────┘
                        │
                   ┌────▼────┐
                   │Reporting│
                   │& Export │
                   └─────────┘
```

### Component Interaction Flow

```
User Input (CSV Data)
        │
        ▼
┌───────────────────┐
│ Data Preparation  │ ← Enrichment utilities
└────────┬──────────┘
         │ (Time series, ratios, features)
         ▼
┌───────────────────┐
│  Model Training   │ ← All banks data for peer benchmarking
└────────┬──────────┘
         │ (Trained models + peer benchmarks + clusters)
         ▼
┌───────────────────┐
│ Risk Assessment   │ ← Expert rules from config
└────────┬──────────┘
         │ (Risk scores + explanations)
         ▼
┌───────────────────┐
│   Reporting       │ ← Templates & formatting
└────────┬──────────┘
         │
         ▼
Output (JSON, Excel, Dashboard, PDF)
```

---

## Project Structure

```
bank-audit-system/
│
├── config/                      # Configuration files
│   ├── __init__.py
│   ├── expert_rules.json        # Expert rule thresholds
│   └── expert_rules_config.py   # Configuration loader
│
├── data/                        # Data storage
│   └── time_series_dataset_enriched_v2.csv  # Primary dataset
│
├── docs/                        # Documentation
│   ├── 00_START_HERE_REPRODUCIBILITY.md
│   ├── PROJECT_INDEX.md
│   ├── REGULATORY_TRANSPARENCY_SUMMARY.md
│   ├── NEW_INDICATORS_GUIDE.md
│   ├── BATCH_PROCESSING_GUIDE.md
│   └── ... (33 documentation files)
│
├── logs/                        # Application logs
│   ├── audit_system_20260110.log
│   └── ...
│
├── outputs/                     # Generated outputs
│   ├── vcb_bank_dashboard.png
│   ├── vcb_bank_report.json
│   └── vcb_bank_report.xlsx
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_imports.py
│   ├── test_exception_handling.py
│   ├── test_critical_calculations.py
│   ├── test_regulatory_transparency.py
│   ├── test_nsfr_integration.py
│   ├── test_nsfr_end_to_end.py
│   ├── test_wholesale_funding_integration.py
│   ├── test_revision.py
│   ├── test_state_management.py
│   └── test_state_simple.py
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── batch_processing.py      # Batch/vectorized operations
│   ├── dataset_enrichment.py    # Dataset enhancement
│   ├── add_new_indicators.py    # Indicator calculation
│   ├── macro_adjustments.py     # Macroeconomic adjustments
│   └── reproducibility.py       # Reproducibility utilities
│
├── 0_example_usage.py           # Usage examples & demo
│
├── 1_data_preparation.py        # Data loading & transformation
├── 2_model_anomaly_detection.py # Anomaly detection ensemble
├── 2_model_base_risk.py         # Base risk model class
├── 2_model_credit_risk.py       # Credit risk assessment
├── 2_model_liquidity_risk.py    # Liquidity risk assessment
│
├── 3_reporting_analysis.py      # Report generation
├── 4_utility_functions.py       # Shared utilities (ratios, etc.)
├── 5_bank_audit_system.py       # Main orchestrator
├── 6_logging_config.py          # Logging configuration
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

---

## Core Components

### 1. Data Preparation (`1_data_preparation.py`)

**Purpose**: Load, validate, and prepare time-series financial data.

**Key Classes**:
- `DataPreparation`: Main class for data loading and transformation

**Responsibilities**:
- Load time series data for individual banks
- Calculate financial ratios across periods
- Compute growth rates (YoY, QoQ)
- Analyze time series trends
- Prepare ML-ready features

**Key Methods**:
```python
load_time_series_data(df, bank_id) -> Dict
calculate_time_series_ratios() -> pd.DataFrame
calculate_growth_rates() -> pd.DataFrame
analyze_time_series_trends() -> Dict
prepare_time_series_features() -> pd.DataFrame
```

---

### 2. Anomaly Detection Model (`2_model_anomaly_detection.py`)

**Purpose**: Detect unusual patterns and potential fraud using ensemble ML.

**Key Classes**:
- `AnomalyDetectionModel`: Ensemble anomaly detector

**ML Algorithms**:
- Isolation Forest
- DBSCAN (density-based clustering)
- One-Class SVM
- Local Outlier Factor (LOF)
- Elliptic Envelope

**Key Innovations**:
- **Ensemble Voting**: Combines predictions from 5 models
- **Peer Clustering**: KMeans-based peer grouping with dynamic k selection
  - Uses silhouette score and Calinski-Harabasz index for optimal k
  - Clusters by size/business model (assets, deposits, loan/deposit ratio, NIM)
  - Cluster-specific peer benchmarks for fair comparisons
- **Expert Red Flags**: Rule-based composite indicators
- **Regulatory Transparency**: Explains top contributing factors and narratives

**Key Methods**:
```python
train_models(training_data, contamination) -> Dict
detect_anomalies(input_data, voting_threshold) -> Dict
detect_fraud_patterns(data) -> Dict
analyze_anomaly_trends(time_series_data) -> Dict
```

**Output Structure**:
```python
{
    'total_records': int,
    'anomalies_detected': int,
    'anomaly_rate': float,
    'anomalies': [
        {
            'index': int,
            'anomaly_score': float,
            'severity': str,  # CRITICAL, HIGH, MEDIUM, LOW
            'explanation': {
                'top_factors': [...],
                'narrative': str,
                'peer_group': int,  # Cluster assignment
                'model_consensus': {...}
            }
        }
    ]
}
```

---

### 3. Credit Risk Model (`2_model_credit_risk.py`)

**Purpose**: Assess credit risk using supervised and unsupervised learning.

**Key Classes**:
- `CreditRiskModel`: Credit risk assessment engine

**ML Algorithms**:
- **Supervised**: RandomForest, GradientBoosting, XGBoost
- **Unsupervised**: Isolation Forest (for unlabeled data)

**Features Analyzed**:
- NPL ratio
- Provision coverage ratio
- Loan growth
- Sector concentration (HHI)
- Top borrower concentration

**Key Methods**:
```python
train_models(training_data, target_column) -> Dict
predict_risk(input_data) -> Dict
assess_risk_indicators(data) -> List[Dict]
```

**Output Structure**:
```python
{
    'credit_risk_score': float,  # 0-100
    'risk_level': str,  # LOW, MEDIUM, HIGH, CRITICAL
    'risk_indicators': [...],
    'explanation': {
        'key_factors': [...],
        'model_insights': {...},
        'narrative': str
    }
}
```

---

### 4. Liquidity Risk Model (`2_model_liquidity_risk.py`)

**Purpose**: Evaluate liquidity position and stress resilience.

**Key Classes**:
- `LiquidityRiskModel`: Liquidity risk assessment engine

**ML Algorithms**:
- Isolation Forest (anomaly detection)
- Linear Regression, RandomForest, GradientBoosting (forecasting)

**Metrics Evaluated**:
- Liquidity Coverage Ratio (LCR)
- Net Stable Funding Ratio (NSFR)
- Loan-to-Deposit Ratio
- Wholesale funding dependency
- Cash flow survival horizon

**Stress Testing Scenarios**:
- Base case
- Moderate stress
- Severe stress

**Key Methods**:
```python
train_models(training_data) -> Dict
predict_risk(input_data) -> Dict
run_liquidity_stress_test(data) -> Dict
forecast_liquidity_position(data, periods) -> Dict
```

---

### 5. Reporting & Analysis (`3_reporting_analysis.py`)

**Purpose**: Generate comprehensive audit reports with regulatory narratives.

**Key Classes**:
- `ReportingAnalysis`: Report generation and formatting

**Report Sections**:
1. **Executive Summary**: Overall risk assessment
2. **Credit Risk**: Detailed credit risk analysis with explanations
3. **Liquidity Risk**: Liquidity position and stress test results
4. **Anomalies**: Detected anomalies with contributing factors
5. **Expert Rule Violations**: Regulatory compliance checks
6. **High-Risk Periods**: Time-based risk identification
7. **Recommendations**: Actionable insights

**Output Formats**:
- JSON (structured data)
- Excel (multi-sheet workbooks)
- HTML (web-ready reports)
- PDF (printable documents)

**Key Methods**:
```python
generate_comprehensive_report(results) -> Dict
export_report(filename, format) -> None
create_dashboard(results, save_path) -> None
```

---

### 6. Utility Functions (`4_utility_functions.py`)

**Purpose**: Centralized financial calculations and shared utilities.

**Key Classes**:
- `FinancialRatioCalculator`: Unified ratio calculations
- `TimeSeriesFeaturePreparation`: ML feature engineering
- `ModelTrainingPipeline`: Generic model training workflows

**Calculated Ratios** (40+ ratios):
- **Credit Risk**: NPL ratio, provision coverage
- **Capital Adequacy**: CAR, equity/assets
- **Liquidity**: LCR, NSFR, loan/deposit
- **Profitability**: ROA, ROE, NIM, cost/income
- **Concentration**: Sector HHI, top borrowers, geographic
- **Off-Balance Sheet**: OBS/loans, derivatives/assets
- **Efficiency**: Employee productivity, asset utilization

**Key Methods**:
```python
FinancialRatioCalculator.calculate_ratios(data) -> Dict
TimeSeriesFeaturePreparation.prepare_training_features(...) -> np.ndarray
ModelTrainingPipeline.train_models_on_features(...) -> Tuple
```

---

### 7. Bank Audit System Orchestrator (`5_bank_audit_system.py`)

**Purpose**: Main orchestrator coordinating all components.

**Key Classes**:
- `BankAuditSystem`: Central audit workflow manager

**Workflow**:
1. Load and prepare data
2. Train all models (using all banks for peer context)
3. Assess risks (anomaly, credit, liquidity)
4. Apply expert rules
5. Identify high-risk periods
6. Generate comprehensive report
7. Export results and dashboards

**Key Methods**:
```python
run_complete_audit(df, bank_id, all_banks_data) -> Dict
load_and_prepare_data(df, bank_id) -> Dict
train_models(all_banks_data) -> Dict
assess_all_risks(prepared_data) -> Dict
apply_expert_rules(prepared_data) -> List[Dict]
identify_high_risk_periods(prepared_data) -> List[Dict]
generate_comprehensive_report() -> Dict
export_results(filename, format) -> None
create_dashboard(report, save_path) -> None
print_summary(report) -> None
```

---

### 8. Logging Configuration (`6_logging_config.py`)

**Purpose**: Structured logging with context tracking.

**Key Classes**:
- `AuditLogger`: Centralized logging manager
- Custom Exceptions: `DataValidationError`, `ModelTrainingError`, etc.

**Features**:
- Structured JSON logging
- Console and file output
- Context injection (bank_id, method names)
- Log rotation
- Error tracking and reporting

**Usage**:
```python
from logging_config import AuditLogger
logger = AuditLogger.get_logger(__name__)
logger.info("Message", extra={'bank_id': 'ABC', 'method': 'train_models'})
```

---

## Data Flow

### Training Phase

```
All Banks CSV Dataset
        │
        ├─> Data Preparation
        │       │
        │       ├─> Ratio Calculation
        │       ├─> Growth Metrics
        │       ├─> Feature Engineering
        │       └─> Enrichment (NIM, OBS, Concentration)
        │
        ├─> Peer Clustering (KMeans)
        │       │
        │       ├─> Feature Selection (assets, deposits, NIM, etc.)
        │       ├─> Dynamic k Selection (silhouette + CH)
        │       └─> Cluster Assignment
        │
        ├─> Anomaly Model Training
        │       │
        │       ├─> Isolation Forest
        │       ├─> DBSCAN
        │       ├─> One-Class SVM
        │       ├─> LOF
        │       ├─> Elliptic Envelope
        │       └─> Cluster-Specific Benchmarks
        │
        ├─> Credit Model Training
        │       │
        │       ├─> RandomForest
        │       ├─> GradientBoosting
        │       └─> XGBoost
        │
        └─> Liquidity Model Training
                │
                ├─> Anomaly Detection
                └─> Forecasting Models
```

### Prediction Phase

```
Target Bank Data
        │
        ├─> Feature Preparation
        │
        ├─> Cluster Assignment (based on bank features)
        │
        ├─> Anomaly Detection
        │       │
        │       ├─> Ensemble Prediction
        │       ├─> Cluster-Specific Benchmarks
        │       └─> Explanation Generation (with peer group)
        │
        ├─> Credit Risk Assessment
        │       │
        │       └─> Risk Scoring + Explanation
        │
        ├─> Liquidity Risk Assessment
        │       │
        │       ├─> Risk Scoring
        │       ├─> Stress Testing
        │       └─> Explanation
        │
        ├─> Expert Rules Validation
        │
        └─> Report Generation
                │
                └─> Output (JSON, Excel, Dashboard, PDF)
```

---

## Key Features

### 1. Regulatory Transparency & Explainability

- **Peer Benchmarking**: Compare metrics against cluster-specific peers
- **Factor Decomposition**: Identify top contributing factors for each risk
- **Regulatory Narratives**: Human-readable explanations for all risk scores
- **Model Consensus**: Show which models flagged each anomaly
- **Deviation Analysis**: Quantify how far metrics deviate from peer averages

### 2. Advanced Peer Clustering

- **Dynamic k Selection**: Automatically determines optimal number of clusters
  - Uses silhouette score (prioritized)
  - Uses Calinski-Harabasz index (tiebreaker)
  - Evaluates k from 2 to min(8, n_banks-1)
- **Business-Model Grouping**: Clusters by size, deposit base, business mix
- **Fair Comparisons**: Compare banks within their peer group only
- **Cluster-Aware Benchmarks**: Separate statistics per cluster

### 3. Ensemble Anomaly Detection

- **5 ML Algorithms**: Isolation Forest, DBSCAN, SVM, LOF, Elliptic Envelope
- **Voting Mechanism**: Configurable consensus threshold
- **Expert Red Flags**: Combines ML with domain rules
- **Severity Classification**: CRITICAL, HIGH, MEDIUM, LOW

### 4. Multi-Dimensional Risk Assessment

- **Credit Risk**: NPL, provisions, concentrations, growth
- **Liquidity Risk**: LCR, NSFR, stress testing, forecasting
- **Operational Risk**: Anomalies, fraud patterns

### 5. Batch Processing & Optimization

- **Vectorized Operations**: 10-100x faster than row-by-row loops
- **Parallel Audits**: Process multiple banks efficiently
- **Aggregation**: Bank-level statistics, rankings, outlier detection

### 6. Reproducibility

- **Fixed Random Seeds**: Ensures consistent results across runs
- **Verification Tools**: Validate reproducibility of ML outputs
- **Audit Trail**: Full logging of all operations

### 7. Dataset Enrichment

- **66 Columns**: Enriched with NIM, OBS, concentration metrics
- **Composite Indicators**: Multi-factor risk signals
- **Synthetic Data Generation**: For testing and demos

---

## Technology Stack

### Core Libraries

- **Python 3.8+**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
  - Ensemble models (IsolationForest, RandomForest, GradientBoosting)
  - Clustering (KMeans, DBSCAN)
  - Metrics (silhouette_score, calinski_harabasz_score)
  - Preprocessing (StandardScaler)
  - Linear models, SVM, neighbors (LOF)
- **XGBoost**: Gradient boosting
- **Matplotlib/Seaborn**: Visualization
- **Openpyxl**: Excel export
- **ReportLab**: PDF generation

### Development Tools

- **Pytest**: Unit testing
- **Logging**: Structured logging
- **Type Hints**: Code clarity and IDE support

---

## Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from bank_audit_system import BankAuditSystem
import pandas as pd

# Load data
df = pd.read_csv('data/time_series_dataset_enriched_v2.csv')

# Initialize audit system
audit = BankAuditSystem(
    bank_name="VCB Bank",
    audit_period="2024-Q1"
)

# Run complete audit
report = audit.run_complete_audit(
    df=df,
    bank_id="VCB",
    all_banks_data=df  # For peer benchmarking and clustering
)

# View summary
audit.print_summary(report)

# Export results
audit.export_results("outputs/report.json", format='json')
audit.create_dashboard(report, save_path="outputs/dashboard.png")
```

---

## Usage Examples

See `0_example_usage.py` for 6 comprehensive examples:

1. **Simple Audit**: One-line audit for a single bank
2. **Step-by-Step Audit**: Detailed control over each phase
3. **Component-Level**: Using individual modules directly
4. **Multiple Banks**: Batch analysis and comparison
5. **Custom Workflow**: Tailored analysis pipelines
6. **Batch Processing**: Optimized vectorized operations

---

## Testing

### Test Suite Location

All tests are in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_regulatory_transparency.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=.
```

### Test Categories

1. **Unit Tests**:
   - `test_imports.py`: Module import validation
   - `test_critical_calculations.py`: Core calculation accuracy
   - `test_exception_handling.py`: Error handling

2. **Integration Tests**:
   - `test_regulatory_transparency.py`: End-to-end transparency validation
   - `test_nsfr_integration.py`: NSFR calculation integration
   - `test_wholesale_funding_integration.py`: Funding ratio integration

3. **End-to-End Tests**:
   - `test_nsfr_end_to_end.py`: Full NSFR workflow

4. **State Management Tests**:
   - `test_state_management.py`: Model state persistence
   - `test_state_simple.py`: Basic state validation

5. **Regression Tests**:
   - `test_revision.py`: Post-refactor validation

---

## Configuration

### Expert Rules (`config/expert_rules.json`)

Define regulatory thresholds:

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
    "max_loan_to_deposit": 0.85,
    "severity": "HIGH"
  },
  "concentration": {
    "max_sector_hhi": 0.20,
    "max_top_borrower_ratio": 0.10,
    "severity": "HIGH"
  }
}
```

### Environment Variables

- `EXPERT_RULES_PATH`: Custom path to expert rules JSON
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

---

## Project Highlights

### Recent Enhancements

1. **Regulatory Transparency** (Phase 3):
   - Added ML explanations across all risk models
   - Peer benchmark comparisons
   - Narrative generation for regulators

2. **Dataset Enrichment** (Phase 3.5):
   - Net Interest Margin (NIM) components
   - Off-balance sheet ratios (OBS, derivatives, guarantees)
   - Concentration metrics (deposit, borrower, geographic)
   - Composite risk indicators

3. **Peer Clustering** (Current):
   - KMeans-based peer grouping
   - Dynamic k selection (silhouette + Calinski-Harabasz)
   - Cluster-aware anomaly detection and benchmarks
   - Fair peer comparisons by bank size/model

4. **Batch Processing** (Optimized):
   - Vectorized operations for large datasets
   - 10-100x performance improvement
   - Parallel audit execution

### Design Principles

- **Modularity**: Each component is independent and reusable
- **Extensibility**: Easy to add new models or risk types
- **Transparency**: Full explainability for regulatory compliance
- **Performance**: Vectorized operations for large-scale audits
- **Reproducibility**: Fixed seeds and audit trails
- **Error Resilience**: Comprehensive exception handling

---

## Future Roadmap

1. **Real-Time Monitoring**: Streaming data integration
2. **Advanced Clustering**: Consider categorical features (bank type, region)
3. **Deep Learning**: Neural networks for time-series forecasting
4. **Interactive Dashboards**: Web-based visualization
5. **API Layer**: RESTful API for integration
6. **Cloud Deployment**: Scalable cloud infrastructure
7. **Multi-Language Support**: Internationalization

---

## Support & Documentation

- **Full Documentation**: See `docs/` folder for detailed guides
- **Getting Started**: `docs/00_START_HERE_REPRODUCIBILITY.md`
- **API Reference**: Component-specific docs in each module
- **Examples**: `0_example_usage.py` for runnable code samples

---

## License

[Specify License]

---

## Contributors

[List Contributors]

---

**Last Updated**: January 10, 2026

**Version**: 1.1.0
