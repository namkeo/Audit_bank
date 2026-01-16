# -*- coding: utf-8 -*-
"""
Mô-đun Báo cáo và Phân tích
Xử lý tạo báo cáo, trực quan hóa và phân tích
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from datetime import datetime
import importlib.util

# Import AuditLogger from 6_logging_config.py
spec = importlib.util.spec_from_file_location("logging_config", "6_logging_config.py")
logging_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_module)
AuditLogger = logging_module.AuditLogger

logger = AuditLogger.get_logger(__name__)


class ReportingAnalysis:
    """
    Mô-đun Báo cáo và Phân tích
    Tạo báo cáo và trực quan hóa toàn diện
    """
    
    def __init__(self, bank_name: str, audit_period: str):
        self.bank_name = bank_name
        self.audit_period = audit_period
        self.reports = {}
        self.visualizations = []
        # Quantile-based classification (populated during batch processing)
        self.all_scores = []  # Collect all scores for quantile calculation
        self.quantiles = None  # Will store (q25, q50, q75) when computed
        self.use_quantile_classification = True  # Enable quantile-based classification
        
    def generate_comprehensive_report(self, 
                                     credit_risk_results: Dict,
                                     liquidity_risk_results: Dict,
                                     anomaly_results: Dict,
                                     time_series_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive audit report
        
        Args:
            credit_risk_results: Results from credit risk assessment
            liquidity_risk_results: Results from liquidity risk assessment
            anomaly_results: Results from anomaly detection
            time_series_data: Historical time series data
            
        Returns:
            Dictionary containing comprehensive report
        """
        report = {
            'bank_name': self.bank_name,
            'audit_period': self.audit_period,
            'report_date': datetime.now().isoformat(),
            'executive_summary': self._generate_executive_summary(
                credit_risk_results,
                liquidity_risk_results,
                anomaly_results
            ),
            'risk_assessments': {
                'credit_risk': self._format_credit_risk_section(credit_risk_results),
                'liquidity_risk': self._format_liquidity_risk_section(liquidity_risk_results),
                'anomaly_detection': self._format_anomaly_section(anomaly_results)
            },
            'time_series_analysis': self._analyze_time_series(time_series_data),
            'recommendations': self._generate_recommendations(
                credit_risk_results,
                liquidity_risk_results,
                anomaly_results
            ),
            'overall_risk_score': self._calculate_overall_risk_score(
                credit_risk_results,
                liquidity_risk_results,
                anomaly_results
            )
        }
        
        self.reports['comprehensive'] = report
        return report
    
    def _generate_executive_summary(self, credit_results: Dict,
                                   liquidity_results: Dict,
                                   anomaly_results: Dict) -> Dict:
        """
        Generate executive summary
        
        Returns:
            Dictionary containing executive summary
        """
        summary = {
            'overall_status': 'UNKNOWN',
            'key_findings': [],
            'critical_issues': [],
            'strengths': []
        }
        
        # Analyze credit risk
        if credit_results and 'credit_risk_score' in credit_results:
            credit_score = credit_results['credit_risk_score']
            credit_level = credit_results.get('risk_level', 'UNKNOWN')
            
            if credit_score > 70:
                summary['critical_issues'].append(
                    f"High credit risk detected (Score: {credit_score:.1f})"
                )
            elif credit_score < 30:
                summary['strengths'].append(
                    f"Strong credit quality (Score: {credit_score:.1f})"
                )
        
        # Analyze liquidity risk
        if liquidity_results and 'liquidity_risk_score' in liquidity_results:
            liquidity_score = liquidity_results['liquidity_risk_score']
            
            if liquidity_score > 70:
                summary['critical_issues'].append(
                    f"High liquidity risk detected (Score: {liquidity_score:.1f})"
                )
            elif liquidity_score < 30:
                summary['strengths'].append(
                    f"Strong liquidity position (Score: {liquidity_score:.1f})"
                )
        
        # Analyze anomalies
        if anomaly_results and 'anomalies_detected' in anomaly_results:
            anomaly_count = anomaly_results['anomalies_detected']
            
            if anomaly_count > 0:
                summary['key_findings'].append(
                    f"{anomaly_count} anomalies detected requiring investigation"
                )
        
        # Determine overall status
        if summary['critical_issues']:
            summary['overall_status'] = 'HIGH_RISK'
        elif summary['key_findings']:
            summary['overall_status'] = 'MEDIUM_RISK'
        else:
            summary['overall_status'] = 'LOW_RISK'
        
        return summary
    
    def _format_credit_risk_section(self, results: Dict) -> Dict:
        """
        Format credit risk section of report with regulatory explanations
        """
        if not results:
            return {'status': 'No data'}
        
        section = {
            'risk_score': results.get('credit_risk_score', 0),
            'risk_level': results.get('risk_level', 'UNKNOWN'),
            'indicators': results.get('indicators', {}),
            'key_metrics': [],
            'regulatory_narrative': results.get('regulatory_narrative', ''),  # NEW
            'ml_explanation': results.get('explanation', {})  # NEW
        }
        
        # Extract key metrics from indicators
        indicators = results.get('indicators', {})
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, dict):
                section['key_metrics'].append({
                    'name': indicator_name,
                    'value': indicator_data.get('value', 0),
                    'status': indicator_data.get('status', 'UNKNOWN'),
                    'description': indicator_data.get('description', '')
                })
        
        return section
    
    def _format_liquidity_risk_section(self, results: Dict) -> Dict:
        """
        Format liquidity risk section of report with regulatory explanations
        """
        if not results:
            return {'status': 'No data'}
        
        section = {
            'risk_score': results.get('liquidity_risk_score', 0),
            'risk_level': results.get('risk_level', 'UNKNOWN'),
            'indicators': results.get('indicators', {}),
            'key_metrics': [],
            'regulatory_narrative': results.get('regulatory_narrative', ''),  # NEW
            'ml_explanation': results.get('explanation', {})  # NEW
        }
        
        # Extract key metrics
        indicators = results.get('indicators', {})
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, dict):
                section['key_metrics'].append({
                    'name': indicator_name,
                    'value': indicator_data.get('value', 0),
                    'status': indicator_data.get('status', 'UNKNOWN'),
                    'description': indicator_data.get('description', '')
                })
        
        return section
    
    def _format_anomaly_section(self, results: Dict) -> Dict:
        """
        Format anomaly detection section of report with regulatory explanations
        """
        if not results:
            return {'status': 'No data'}
        
        section = {
            'total_anomalies': results.get('anomalies_detected', 0),
            'anomaly_rate': results.get('anomaly_rate', 0),
            'anomalies': results.get('anomalies', [])
        }
        
        # Categorize anomalies by severity
        if 'anomalies' in results:
            severity_counts = {}
            anomalies_with_explanations = []  # NEW
            
            for anomaly in results['anomalies']:
                severity = anomaly.get('severity', 'UNKNOWN')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Extract explanation for each anomaly (NEW)
                if 'explanation' in anomaly:
                    anomalies_with_explanations.append({
                        'index': anomaly.get('index', -1),
                        'severity': severity,
                        'score': anomaly.get('anomaly_score', 0),
                        'regulatory_narrative': anomaly['explanation'].get('narrative', ''),
                        'top_factors': anomaly['explanation'].get('top_factors', [])
                    })
            
            section['severity_distribution'] = severity_counts
            section['explained_anomalies'] = anomalies_with_explanations  # NEW
        
        return section
    
    def _analyze_time_series(self, data: pd.DataFrame) -> Dict:
        """
        Analyze time series trends
        """
        if data.empty:
            return {'status': 'No data'}
        
        analysis = {
            'periods_analyzed': len(data),
            'trends': {},
            'high_risk_periods': []
        }
        
        # Analyze numeric columns for trends
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'period':
                values = data[col].values
                
                if len(values) >= 2:
                    # Calculate trend
                    slope = np.polyfit(range(len(values)), values, 1)[0]
                    
                    analysis['trends'][col] = {
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'slope': float(slope),
                        'latest_value': float(values[-1]) if len(values) > 0 else 0,
                        'change_pct': float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
        
        return analysis
    
    def _generate_recommendations(self, credit_results: Dict,
                                 liquidity_results: Dict,
                                 anomaly_results: Dict) -> List[Dict]:
        """
        Generate actionable recommendations
        """
        recommendations = []
        
        # Credit risk recommendations
        if credit_results and 'credit_risk_score' in credit_results:
            credit_score = credit_results['credit_risk_score']
            
            if credit_score > 70:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Credit Risk',
                    'description': 'Implement enhanced credit monitoring and loan review procedures',
                    'action': 'Review loan portfolio quality and increase loan loss provisions',
                    'timeline': 'Immediate'
                })
            
            # Check NPL ratio
            indicators = credit_results.get('indicators', {})
            if 'npl_ratio' in indicators:
                npl_data = indicators['npl_ratio']
                if npl_data.get('status') == 'HIGH_RISK':
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Asset Quality',
                        'description': 'Non-performing loan ratio exceeds acceptable threshold',
                        'action': 'Implement aggressive NPL recovery program and strengthen underwriting',
                        'timeline': '30 days'
                    })
        
        # Liquidity risk recommendations
        if liquidity_results and 'liquidity_risk_score' in liquidity_results:
            liquidity_score = liquidity_results['liquidity_risk_score']
            
            if liquidity_score > 70:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Liquidity Risk',
                    'description': 'Liquidity position requires immediate attention',
                    'action': 'Increase liquid assets and reduce loan-to-deposit ratio',
                    'timeline': 'Immediate'
                })
        
        # Anomaly recommendations
        if anomaly_results and anomaly_results.get('anomalies_detected', 0) > 0:
            critical_anomalies = [a for a in anomaly_results.get('anomalies', []) 
                                if a.get('severity') in ['CRITICAL', 'HIGH']]
            
            if critical_anomalies:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Anomaly Detection',
                    'description': f'{len(critical_anomalies)} high-severity anomalies detected',
                    'action': 'Conduct detailed investigation of flagged transactions and activities',
                    'timeline': '7 days'
                })
        
        # General recommendations if no issues
        if not recommendations:
            recommendations.append({
                'priority': 'LOW',
                'category': 'General',
                'description': 'Continue monitoring key risk indicators',
                'action': 'Maintain current risk management practices',
                'timeline': 'Ongoing'
            })
        
        return recommendations
    
    def _calculate_overall_risk_score(self, credit_results: Dict,
                                     liquidity_results: Dict,
                                     anomaly_results: Dict) -> Dict:
        """
        Calculate overall composite risk score
        """
        scores = []
        weights = []
        
        # Credit risk (40% weight)
        if credit_results and 'credit_risk_score' in credit_results:
            scores.append(credit_results['credit_risk_score'])
            weights.append(0.4)
        
        # Liquidity risk (35% weight)
        if liquidity_results and 'liquidity_risk_score' in liquidity_results:
            scores.append(liquidity_results['liquidity_risk_score'])
            weights.append(0.35)
        
        # Anomaly score (25% weight)
        if anomaly_results and 'anomaly_rate' in anomaly_results:
            anomaly_score = anomaly_results['anomaly_rate'] * 100
            scores.append(anomaly_score)
            weights.append(0.25)
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            base_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

            # Count early warning signals across risk areas (if provided)
            warning_count = 0
            for res in (credit_results, liquidity_results, anomaly_results):
                if res and 'early_warning_signals' in res:
                    warning_count += len(res.get('early_warning_signals') or [])

            # Apply conservative bump for multiple warnings (management attention)
            if warning_count >= 3:
                warning_adjustment = 10.0
            elif warning_count >= 1:
                warning_adjustment = 5.0
            else:
                warning_adjustment = 0.0

            overall_score = min(100.0, base_score + warning_adjustment)

            # Classify risk level (use adjusted score); escalate one level if many warnings
            risk_level = self._classify_level(overall_score)
            if warning_count >= 3 and risk_level != "CRITICAL":
                risk_level = self._escalate_level(risk_level)
            
            return {
                'overall_score': overall_score,
                'base_score': base_score,
                'risk_level': risk_level,
                'early_warning': {
                    'count': warning_count,
                    'adjustment': warning_adjustment,
                    'applied': warning_adjustment > 0
                },
                'component_scores': {
                    'credit': credit_results.get('credit_risk_score', 0) if credit_results else 0,
                    'liquidity': liquidity_results.get('liquidity_risk_score', 0) if liquidity_results else 0,
                    'anomaly': anomaly_results.get('anomaly_rate', 0) * 100 if anomaly_results else 0
                }
            }
        
        return {'overall_score': 0, 'risk_level': 'UNKNOWN'}
    
    def _classify_level(self, score: float) -> str:
        """
        Classify risk level using quantile-based method (adaptive) or fixed thresholds.
        
        Quantile Method (when quantiles available):
            MINIMAL: score <= Q1 (25th percentile) - Bottom 25% (best performers)
            LOW:     Q1 < score <= Q2 (50th percentile) - 25-50th percentile
            MEDIUM:  Q2 < score <= Q3 (75th percentile) - 50-75th percentile
            HIGH:    score > Q3 - Top 25% (worst performers)
        
        Fixed Threshold Method (fallback):
            MINIMAL: score < 30
            LOW:     30 <= score < 50
            MEDIUM:  50 <= score < 70
            HIGH:    score >= 70
        """
        # Use quantile-based classification if quantiles are available
        if self.use_quantile_classification and self.quantiles is not None:
            q25, q50, q75 = self.quantiles
            if score <= q25:
                return "MINIMAL"
            elif score <= q50:
                return "LOW"
            elif score <= q75:
                return "MEDIUM"
            else:
                return "HIGH"
        
        # Fallback to fixed thresholds
        if score >= 70:
            return "HIGH"
        elif score >= 50:
            return "MEDIUM"
        elif score >= 30:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _escalate_level(self, level: str) -> str:
        """Escalate one level up when multiple early warnings exist."""
        order = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if level in order:
            idx = min(len(order) - 1, order.index(level) + 1)
            return order[idx]
        return level
    
    def set_quantile_thresholds(self, quantiles: tuple):
        """
        Set quantile thresholds for classification.
        
        Args:
            quantiles: Tuple of (q25, q50, q75) percentile values
        """
        self.quantiles = quantiles
        logger.info(f"Quantile thresholds set: Q1={quantiles[0]:.2f}, Q2={quantiles[1]:.2f}, Q3={quantiles[2]:.2f}")
    
    def compute_quantiles_from_scores(self):
        """
        Compute quantile thresholds from collected scores.
        Call this after collecting all individual bank scores.
        """
        if len(self.all_scores) < 4:
            logger.warning("Not enough scores to compute meaningful quantiles")
            return
        
        import numpy as np
        q25 = np.quantile(self.all_scores, 0.25)
        q50 = np.quantile(self.all_scores, 0.50)
        q75 = np.quantile(self.all_scores, 0.75)
        
        self.set_quantile_thresholds((q25, q50, q75))
    
    def add_score_to_collection(self, score: float):
        """
        Add a score to the collection for quantile calculation.
        
        Args:
            score: Overall risk score to add to collection
        """
        self.all_scores.append(score)
    
    def print_report(self, report: Optional[Dict] = None):
        """
        Print formatted report to console
        
        Args:
            report: Report dictionary (uses comprehensive report if not provided)
        """
        if report is None:
            report = self.reports.get('comprehensive', {})

        # Inline helper to show early warning impact if present
        early_warning = report.get('overall_risk_score', {}).get('early_warning', {})
        
        if not report:
            logger.warning("No report available")
            return
        
        logger.info(f"BANK AUDIT REPORT - {report.get('bank_name', 'Unknown')}")
        logger.info(f"Audit Period: {report.get('audit_period', 'Unknown')}")
        logger.info(f"Report Date: {report.get('report_date', 'Unknown')}")
        
        # Executive Summary
        if 'executive_summary' in report:
            summary = report['executive_summary']
            logger.info("EXECUTIVE SUMMARY")
            logger.info(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
            
            if summary.get('critical_issues'):
                logger.warning("Critical Issues:")
                for issue in summary['critical_issues']:
                    logger.warning(f"  • {issue}")
            
            if summary.get('key_findings'):
                logger.info("Key Findings:")
                for finding in summary['key_findings']:
                    logger.info(f"  • {finding}")
            
            if summary.get('strengths'):
                logger.info("Strengths:")
                for strength in summary['strengths']:
                    logger.info(f"  • {strength}")
        
        # Overall Risk Score
        if 'overall_risk_score' in report:
            risk_score = report['overall_risk_score']
            logger.info("OVERALL RISK ASSESSMENT")
            logger.info(f"Risk Score: {risk_score.get('overall_score', 0):.2f}")
            logger.info(f"Risk Level: {risk_score.get('risk_level', 'UNKNOWN')}")
            ew = risk_score.get('early_warning', {})
            if ew.get('applied'):
                logger.info(f"Early Warnings: {ew.get('count', 0)} (applied +{ew.get('adjustment', 0):.1f} to score)")
            elif ew:
                logger.info(f"Early Warnings: {ew.get('count', 0)} (no adjustment)")
        
        # Recommendations
        if 'recommendations' in report:
            logger.info("RECOMMENDATIONS")
            for i, rec in enumerate(report['recommendations'], 1):
                logger.info(f"{i}. [{rec.get('priority', 'LOW')}] {rec.get('description', '')}")
                logger.info(f"   Action: {rec.get('action', '')}")
                logger.info(f"   Timeline: {rec.get('timeline', 'N/A')}")
    
    def create_risk_dashboard(self, report: Dict, save_path: Optional[str] = None):
        """
        Create visual dashboard of risk metrics
        
        Args:
            report: Report dictionary
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Risk Dashboard - {self.bank_name}", fontsize=16, fontweight='bold')
        
        # Overall Risk Score Gauge
        ax1 = axes[0, 0]
        if 'overall_risk_score' in report:
            score = report['overall_risk_score'].get('overall_score', 0)
            self._plot_gauge(ax1, score, "Overall Risk Score")
        
        # Component Risk Scores
        ax2 = axes[0, 1]
        if 'overall_risk_score' in report:
            components = report['overall_risk_score'].get('component_scores', {})
            if components:
                self._plot_component_scores(ax2, components)
        
        # Risk Level Distribution
        ax3 = axes[1, 0]
        if 'risk_assessments' in report:
            self._plot_risk_levels(ax3, report['risk_assessments'])
        
        # Anomaly Distribution
        ax4 = axes[1, 1]
        if 'risk_assessments' in report and 'anomaly_detection' in report['risk_assessments']:
            anomaly_data = report['risk_assessments']['anomaly_detection']
            if 'severity_distribution' in anomaly_data:
                self._plot_anomaly_distribution(ax4, anomaly_data['severity_distribution'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to: {save_path}")
        else:
            plt.show()
    
    def _plot_gauge(self, ax, value: float, title: str):
        """Plot a gauge chart"""
        # Color based on value
        if value >= 70:
            color = 'red'
        elif value >= 50:
            color = 'orange'
        elif value >= 30:
            color = 'yellow'
        else:
            color = 'green'
        
        ax.barh(0, value, color=color, height=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Risk Score')
        ax.text(value/2, 0, f'{value:.1f}', ha='center', va='center', 
               fontweight='bold', fontsize=12, color='white')
        ax.set_yticks([])
    
    def _plot_component_scores(self, ax, components: Dict):
        """Plot component risk scores"""
        labels = list(components.keys())
        values = list(components.values())
        
        colors = ['#ff6b6b' if v >= 70 else '#ffd93d' if v >= 50 else '#6bcf7f' 
                 for v in values]
        
        ax.barh(labels, values, color=colors)
        ax.set_xlabel('Risk Score')
        ax.set_title('Risk Component Scores', fontweight='bold')
        ax.set_xlim(0, 100)
        
        for i, v in enumerate(values):
            ax.text(v + 2, i, f'{v:.1f}', va='center')
    
    def _plot_risk_levels(self, ax, risk_assessments: Dict):
        """Plot risk levels across categories"""
        categories = []
        levels = []
        
        for category, data in risk_assessments.items():
            if isinstance(data, dict) and 'risk_level' in data:
                categories.append(category.replace('_', ' ').title())
                levels.append(data['risk_level'])
        
        if categories:
            level_colors = {
                'CRITICAL': 'darkred',
                'HIGH': 'red',
                'MEDIUM': 'orange',
                'LOW': 'yellow',
                'MINIMAL': 'green',
                'UNKNOWN': 'gray'
            }
            
            colors = [level_colors.get(level, 'gray') for level in levels]
            
            ax.barh(categories, [1]*len(categories), color=colors)
            ax.set_title('Risk Levels by Category', fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            
            for i, level in enumerate(levels):
                ax.text(0.5, i, level, ha='center', va='center', 
                       fontweight='bold', color='white')
    
    def _plot_anomaly_distribution(self, ax, severity_dist: Dict):
        """Plot anomaly severity distribution"""
        if not severity_dist:
            ax.text(0.5, 0.5, 'No anomalies detected', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        labels = list(severity_dist.keys())
        sizes = list(severity_dist.values())
        
        colors = {
            'CRITICAL': 'darkred',
            'HIGH': 'red',
            'MEDIUM': 'orange',
            'LOW': 'yellow'
        }
        
        pie_colors = [colors.get(label, 'gray') for label in labels]
        
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
              startangle=90)
        ax.set_title('Anomaly Severity Distribution', fontweight='bold')
