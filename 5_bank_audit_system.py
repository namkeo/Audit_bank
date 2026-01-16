# -*- coding: utf-8 -*-
"""
Hệ thống Kiểm toán Ngân hàng - Bộ điều phối chính
Phối hợp tất cả các thành phần của hệ thống kiểm toán ngân hàng
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

# Import all modules
import importlib.util
import sys
import os

# Helper function to import modules with numeric prefixes
def _import_module(module_path, class_name):
    """Nhập mô-đun theo đường dẫn tệp và trả về lớp được chỉ định
    
    Args:
        module_path (str): Đường dẫn tệp mô-đun
        class_name (str): Tên lớp cần nhập
        
    Returns:
        Lớp được chỉ định từ mô-đun
        
    Raises:
        ImportError: Nếu không tìm thấy mô-đun hoặc lớp
    """
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Import logging configuration
try:
    AuditLogger = _import_module("6_logging_config.py", "AuditLogger")
except Exception:
    import logging
    class AuditLogger:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)

# Import classes from modules with numeric prefixes
DataPreparation = _import_module("1_data_preparation.py", "DataPreparation")
CreditRiskModel = _import_module("2_model_credit_risk.py", "CreditRiskModel")
LiquidityRiskModel = _import_module("2_model_liquidity_risk.py", "LiquidityRiskModel")
AnomalyDetectionModel = _import_module("2_model_anomaly_detection.py", "AnomalyDetectionModel")
ReportingAnalysis = _import_module("3_reporting_analysis.py", "ReportingAnalysis")

# Import consolidated utility functions and standalone functions
FinancialRatioCalculator = _import_module("4_utility_functions.py", "FinancialRatioCalculator")

# Import standalone utility functions
def _import_standalone_function(module_path, func_name):
    """Import a function from a module"""
    spec = importlib.util.spec_from_file_location("temp_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

identify_high_risk_periods_standalone = _import_standalone_function("4_utility_functions.py", "identify_high_risk_periods")


class BankAuditSystem:
    """
    Bộ điều phối chính cho Hệ thống Kiểm toán Ngân hàng
    Phối hợp chuẩn bị dữ liệu, đánh giá rủi ro và báo cáo
    """
    
    def __init__(self, bank_name: str, audit_period: str):
        """
        Khởi tạo Hệ thống Kiểm toán Ngân hàng
        
        Args:
            bank_name (str): Tên của ngân hàng được kiểm toán
            audit_period (str): Kỳ kiểm toán (ví dụ: "2024-Q1")
            
        Returns:
            None
        """
        self.bank_name = bank_name
        self.audit_period = audit_period
        self.logger = AuditLogger.get_logger(__name__)
        
        self.logger.info(f"Initializing Bank Audit System for {bank_name} - Period: {audit_period}")
        
        # Initialize all components
        self.data_prep = DataPreparation()
        self.credit_risk = CreditRiskModel()
        self.liquidity_risk = LiquidityRiskModel()
        self.anomaly_detection = AnomalyDetectionModel()
        self.reporting = ReportingAnalysis(bank_name, audit_period)
        
        # Store results
        self.results = {}
        self.expert_rules_violations = []
        self.high_risk_periods = []
        
    def initialize_expert_rules(self) -> Dict:
        """
        Khởi tạo các quy tắc chuyên gia và ngưỡng quy định
        
        Returns:
            dict: Từ điển các quy tắc chuyên gia với các ngưỡng kiểm soát rủi ro
            
        Raises:
            FileNotFoundError: Nếu tệp cấu hình quy tắc không tìm thấy
            JSONDecodeError: Nếu tệp cấu hình không hợp lệ
        """
        # Load expert rules from external JSON config (fallback to defaults)
        from config.expert_rules_config import load_expert_rules
        return load_expert_rules()
        """try:
           
        except Exception:
            # Fallback: inline defaults matching Basel III guidance
            return {
                'capital_adequacy': {
                    'min_car': 0.105,
                    'target_car': 0.125,
                    'severity': 'CRITICAL'
                },
                'asset_quality': {
                    'max_npl_ratio': 0.03,
                    'min_coverage_ratio': 0.70,
                    'severity': 'HIGH'
                },
                'liquidity': {
                    'min_lcr': 1.0,
                    'max_loan_to_deposit': 0.85,
                    'severity': 'HIGH'
                },
                'profitability': {
                    'min_roa': 0.005,
                    'min_roe': 0.08,
                    'severity': 'MEDIUM'
                },
                'growth_limits': {
                    'max_loan_growth': 0.30,
                    'max_asset_growth': 0.25,
                    'severity': 'MEDIUM'
                }
            }
    """
    def load_and_prepare_data(self, df: pd.DataFrame, bank_id: str,
                             time_column: str = 'period') -> Dict:
        """
        Tải và chuẩn bị dữ liệu để phân tích
        
        Args:
            df (pd.DataFrame): Dataframe đầu vào
            bank_id (str): Mã nhận dạng ngân hàng
            time_column (str): Tên cột thời gian (mặc định: 'period')
            
        Returns:
            dict: Từ điển chứa dữ liệu đã chuẩn bị (thời gian chuỗi, tỷ lệ, đặc trưng, dữ liệu thô)
            
        Raises:
            ValueError: Nếu bank_id không tìm thấy trong dữ liệu
            KeyError: Nếu các cột bắt buộc bị thiếu
        """
        try:
            self.logger.info(f"Loading data for bank {bank_id}")
            
            # Load time series data
            time_series_data = self.data_prep.load_time_series_data(df, bank_id, time_column)
            
            # Store raw bank data for anomaly detection
            bank_data_raw = df[df['bank_id'] == bank_id].copy() if 'bank_id' in df.columns else df.copy()
            
            # Calculate ratios
            self.logger.info("Calculating financial ratios")
            ratios_df = self.data_prep.calculate_time_series_ratios()
            
            # Prepare features
            self.logger.info("Preparing features for modeling")
            features_df = self.data_prep.prepare_time_series_features()
            
            # Store results
            self.results['time_series_data'] = time_series_data
            self.results['ratios'] = ratios_df
            self.results['features'] = features_df
            self.results['raw_data'] = bank_data_raw  # Raw enriched data for anomaly detection
            
            self.logger.info(f"Data preparation complete for {bank_id}")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error loading and preparing data for {bank_id}: {str(e)}", 
                            extra={'bank_id': bank_id, 'method': 'load_and_prepare_data'})
            raise
    
    def train_models(self, all_banks_data: pd.DataFrame) -> Dict:
        """
        Đào tạo tất cả mô hình rủi ro trên dữ liệu ngành
        
        Args:
            all_banks_data: Dataframe chứa dữ liệu từ tất cả ngân hàng
            
        Returns:
            dict: Từ điển kết quả huấn luyện cho rủi ro tín dụng, thanh khoản và bất thường
            
        Raises:
            ValueError: Nếu dữ liệu huấn luyện không đủ
            RuntimeError: Nếu quá trình huấn luyện thất bại
        """
        try:
            self.logger.info("Training risk assessment models")
            
            training_results = {}
            
            # Prepare features for training (if needed)
            # Note: Using provided all_banks_data directly for now
            
            # Train credit risk models
            self.logger.debug("  - Training credit risk models...")
            credit_features = all_banks_data.copy()
            credit_training = self.credit_risk.train_models(credit_features)
            training_results['credit_risk'] = credit_training
            
            # Train liquidity risk models
            self.logger.debug("  - Training liquidity risk models...")
            liquidity_training = self.liquidity_risk.train_models(credit_features)
            training_results['liquidity_risk'] = liquidity_training
            
            # Train anomaly detection models with same raw features
            # Note: Anomaly detection will prepare its own features internally
            self.logger.debug("  - Training anomaly detection models...")
            anomaly_training = self.anomaly_detection.train_models(credit_features)
            training_results['anomaly_detection'] = anomaly_training
            
            self.results['training'] = training_results
            return training_results
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}", extra={'method': 'train_models'})
            raise
    
    def assess_all_risks(self, prepared_data: Dict) -> Dict:
        """
        Thực hiện đánh giá rủi ro toàn diện
        
        Args:
            prepared_data (dict): Dữ liệu đã chuẩn bị từ load_and_prepare_data
            
        Returns:
            dict: Từ điển chứa kết quả đánh giá rủi ro tín dụng, thanh khoản, bất thường
            
        Raises:
            RuntimeError: Nếu mô hình chưa được huấn luyện
            ValueError: Nếu dữ liệu đầu vào không hợp lệ
        """
        self.logger.info("Performing comprehensive risk assessment...")
        
        risk_assessments = {}
        features_df = prepared_data['features']
        raw_data = prepared_data.get('raw_data', features_df)  # Get raw enriched data for anomaly detection
        
        # Credit Risk Assessment
        self.logger.debug("  - Assessing credit risk...")
        credit_results = self.credit_risk.predict_risk(features_df)
        credit_indicators = self.credit_risk.assess_risk_indicators(features_df)
        risk_assessments['credit_risk'] = credit_results
        
        # Liquidity Risk Assessment
        self.logger.debug("  - Assessing liquidity risk...")
        liquidity_results = self.liquidity_risk.predict_risk(features_df)
        liquidity_indicators = self.liquidity_risk.assess_risk_indicators(features_df)
        risk_assessments['liquidity_risk'] = liquidity_results
        
        # Anomaly Detection - use raw enriched data, same format as training
        self.logger.debug("  - Detecting anomalies...")
        anomaly_results = self.anomaly_detection.detect_anomalies(raw_data)
        fraud_patterns = self.anomaly_detection.detect_fraud_patterns(raw_data)
        risk_assessments['anomaly_detection'] = anomaly_results
        risk_assessments['fraud_patterns'] = fraud_patterns
        
        self.results['risk_assessments'] = risk_assessments
        return risk_assessments
    
    def apply_expert_rules(self, prepared_data: Dict) -> List[Dict]:
        """
        Áp dụng các quy tắc chuyên gia để xác định vi phạm
        
        Args:
            prepared_data (dict): Dữ liệu đã chuẩn bị
            
        Returns:
            list: Danh sách các vi phạm quy tắc được phát hiện
            
        Raises:
            KeyError: Nếu các quy tắc chuyên gia không được khởi tạo
        """
        self.logger.debug("Applying expert rules validation...")
        
        violations = []
        expert_rules = self.initialize_expert_rules()
        
        ratios_df = prepared_data['ratios']
        
        if ratios_df.empty:
            return violations
        
        latest_ratios = ratios_df.iloc[-1]
        
        # Check capital adequacy
        if 'capital_adequacy_ratio' in latest_ratios:
            car = latest_ratios['capital_adequacy_ratio']
            if car < expert_rules['capital_adequacy']['min_car']:
                violations.append({
                    'rule': 'Capital Adequacy',
                    'metric': 'capital_adequacy_ratio',
                    'value': car,
                    'threshold': expert_rules['capital_adequacy']['min_car'],
                    'severity': expert_rules['capital_adequacy']['severity'],
                    'message': f"CAR ({car:.2%}) below minimum requirement ({expert_rules['capital_adequacy']['min_car']:.2%})"
                })
        
        # Check NPL ratio
        if 'npl_ratio' in latest_ratios:
            npl = latest_ratios['npl_ratio']
            if npl > expert_rules['asset_quality']['max_npl_ratio']:
                violations.append({
                    'rule': 'Asset Quality',
                    'metric': 'npl_ratio',
                    'value': npl,
                    'threshold': expert_rules['asset_quality']['max_npl_ratio'],
                    'severity': expert_rules['asset_quality']['severity'],
                    'message': f"NPL ratio ({npl:.2%}) exceeds maximum ({expert_rules['asset_quality']['max_npl_ratio']:.2%})"
                })
        
        # Check liquidity
        if 'loan_to_deposit' in latest_ratios:
            ltd = latest_ratios['loan_to_deposit']
            if ltd > expert_rules['liquidity']['max_loan_to_deposit']:
                violations.append({
                    'rule': 'Liquidity',
                    'metric': 'loan_to_deposit',
                    'value': ltd,
                    'threshold': expert_rules['liquidity']['max_loan_to_deposit'],
                    'severity': expert_rules['liquidity']['severity'],
                    'message': f"Loan-to-deposit ratio ({ltd:.2%}) exceeds maximum ({expert_rules['liquidity']['max_loan_to_deposit']:.2%})"
                })
        
        # Check profitability
        if 'roa' in latest_ratios:
            roa = latest_ratios['roa']
            if roa < expert_rules['profitability']['min_roa']:
                violations.append({
                    'rule': 'Profitability',
                    'metric': 'roa',
                    'value': roa,
                    'threshold': expert_rules['profitability']['min_roa'],
                    'severity': expert_rules['profitability']['severity'],
                    'message': f"ROA ({roa:.2%}) below minimum ({expert_rules['profitability']['min_roa']:.2%})"
                })
        
        self.expert_rules_violations = violations
        return violations
    
    def identify_high_risk_periods(self, prepared_data: Dict) -> List[Dict]:
        """
        Identify periods with elevated risk.
        
        This method delegates to the standalone utility function identify_high_risk_periods()
        to maintain consistency across the codebase. Both approaches yield identical results.
        
        Args:
            prepared_data: Dictionary containing prepared data with 'features' DataFrame
            
        Returns:
            List of dictionaries containing high-risk periods with severity and indicators
        """
        self.logger.debug("Identifying high risk periods...")
        
        # Call standalone utility function (can also be used independently)
        high_risk_periods = identify_high_risk_periods_standalone(
            prepared_data,
            npl_threshold=0.05,
            liquidity_threshold=0.3,
            capital_threshold=0.08,
            min_risk_indicators=2,
            time_column='period'
        )
        
        # Store in instance for later retrieval
        self.high_risk_periods = high_risk_periods
        return high_risk_periods
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive audit report
        
        Returns:
            Complete audit report
        """
        self.logger.debug("Generating comprehensive report...")
        
        # Get all results
        risk_assessments = self.results.get('risk_assessments', {})
        
        credit_results = risk_assessments.get('credit_risk', {})
        liquidity_results = risk_assessments.get('liquidity_risk', {})
        anomaly_results = risk_assessments.get('anomaly_detection', {})
        
        prepared_data = self.results.get('prepared_data', {})
        time_series_data = prepared_data.get('features', pd.DataFrame())
        
        # Generate report
        report = self.reporting.generate_comprehensive_report(
            credit_results,
            liquidity_results,
            anomaly_results,
            time_series_data
        )
        
        # Add expert rules violations
        report['expert_rules_violations'] = self.expert_rules_violations
        
        # Add high risk periods
        report['high_risk_periods'] = self.high_risk_periods
        
        return report
    
    def run_complete_audit(self, df: pd.DataFrame, bank_id: str,
                          all_banks_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Chạy quy trình kiểm toán hoàn chỉnh
        
        Args:
            df (pd.DataFrame): Dataframe của ngân hàng cụ thể
            bank_id (str): Mã nhận dạng ngân hàng
            all_banks_data (Optional[pd.DataFrame]): Dữ liệu từ tất cả ngân hàng để huấn luyện
            
        Returns:
            dict: Từ điển chứa kết quả kiểm toán hoàn chỉnh (rủi ro, báo cáo, vi phạm)
            
        Raises:
            ValueError: Nếu bank_id không hợp lệ
            RuntimeError: Nếu quá trình kiểm toán thất bại
        """
        self.logger.info(f"Starting Complete Audit for {bank_id}")
        
        # Step 1: Load and prepare data
        prepared_data = self.load_and_prepare_data(df, bank_id)
        
        # Step 2: Train models (if industry data provided)
        if all_banks_data is not None:
            self.train_models(all_banks_data)
        
        # Step 3: Assess all risks
        risk_assessments = self.assess_all_risks(prepared_data)
        
        # Step 4: Apply expert rules
        violations = self.apply_expert_rules(prepared_data)
        
        # Step 5: Identify high risk periods
        high_risk = self.identify_high_risk_periods(prepared_data)
        
        # Step 6: Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        self.logger.info("Audit Complete")
        
        return report
    
    def print_summary(self, report: Optional[Dict] = None):
        """
        Print audit summary
        
        Args:
            report: Report to print (uses last generated if not provided)
        """
        if report is None:
            report = self.reporting.reports.get('comprehensive', {})
        
        self.reporting.print_report(report)
    
    def create_dashboard(self, report: Optional[Dict] = None, 
                        save_path: Optional[str] = None):
        """
        Create visual dashboard
        
        Args:
            report: Report to visualize
            save_path: Path to save dashboard
        """
        if report is None:
            report = self.reporting.reports.get('comprehensive', {})
        
        self.reporting.create_risk_dashboard(report, save_path)
    
    def export_results(self, filepath: str, format: str = 'json'):
        """
        Export audit results to file
        
        Args:
            filepath: Path to save file
            format: Export format ('json', 'csv', 'excel')
        """
        import json
        
        report = self.reporting.reports.get('comprehensive', {})
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            self.logger.info(f"Results exported to {filepath}")
        
        elif format == 'excel':
            # Export to Excel with multiple sheets, fallback to CSV if engine missing
            try:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    summary_df = pd.DataFrame([{
                        'Bank': report.get('bank_name', ''),
                        'Audit Period': report.get('audit_period', ''),
                        'Overall Risk Score': report.get('overall_risk_score', {}).get('overall_score', 0),
                        'Risk Level': report.get('overall_risk_score', {}).get('risk_level', '')
                    }])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                    if self.expert_rules_violations:
                        violations_df = pd.DataFrame(self.expert_rules_violations)
                        violations_df.to_excel(writer, sheet_name='Violations', index=False)

                self.logger.info(f"Results exported to {filepath}")
            except ModuleNotFoundError:
                # Fallback: export CSVs when openpyxl is unavailable
                base = filepath[:-5] if filepath.lower().endswith('.xlsx') else filepath
                summary_df = pd.DataFrame([{
                    'Bank': report.get('bank_name', ''),
                    'Audit Period': report.get('audit_period', ''),
                    'Overall Risk Score': report.get('overall_risk_score', {}).get('overall_score', 0),
                    'Risk Level': report.get('overall_risk_score', {}).get('risk_level', '')
                }])
                summary_csv = f"{base}_summary.csv"
                summary_df.to_csv(summary_csv, index=False)

                if self.expert_rules_violations:
                    violations_df = pd.DataFrame(self.expert_rules_violations)
                    violations_csv = f"{base}_violations.csv"
                    violations_df.to_csv(violations_csv, index=False)

                self.logger.warning(f"openpyxl not found; exported CSVs to {summary_csv} and violations (if any)")
        
        else:
            self.logger.error(f"Format '{format}' not supported")
