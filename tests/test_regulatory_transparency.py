# -*- coding: utf-8 -*-
"""
Test script to verify regulatory transparency features
"""

import pandas as pd
import json
from pathlib import Path

# Import AuditLogger
import importlib.util
spec = importlib.util.spec_from_file_location("logging_config", "6_logging_config.py")
logging_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging_module)
AuditLogger = logging_module.AuditLogger

logger = AuditLogger.get_logger(__name__)

# Import bank audit system
spec = importlib.util.spec_from_file_location("bank_audit_system", "5_bank_audit_system.py")
bas_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bas_module)
BankAuditSystem = bas_module.BankAuditSystem


def test_regulatory_explanations():
    """Test that regulatory explanations are generated for all ML outputs"""
    
    print("\n" + "="*80)
    print("REGULATORY TRANSPARENCY TEST")
    print("="*80)
    
    # Load data
    logger.info("Loading dataset...")
    df = pd.read_csv('time_series_dataset_enriched.csv')
    
    # Initialize system
    logger.info("Initializing Bank Audit System for VCB")
    audit_system = BankAuditSystem(bank_name='VCB', audit_period='2024-Q1')
    
    # Run complete audit
    logger.info("Running complete audit...")
    results = audit_system.run_complete_audit(
        df=df,
        bank_id='VCB',
        all_banks_data=df
    )
    
    # Check if explanations exist in anomaly detection results
    print("\n1. ANOMALY DETECTION EXPLANATIONS")
    print("-" * 80)
    
    # Results come from comprehensive report
    if 'risk_assessments' in results and 'anomaly_detection' in results['risk_assessments']:
        anom_section = results['risk_assessments']['anomaly_detection']
        
        if 'explained_anomalies' in anom_section and len(anom_section['explained_anomalies']) > 0:
            print(f"✓ Found {len(anom_section['explained_anomalies'])} explained anomalies")
            
            # Check first anomaly for explanation
            first_anomaly = anom_section['explained_anomalies'][0]
            
            if 'regulatory_narrative' in first_anomaly:
                print(f"\nRegulatory Narrative:")
                print(f"  {first_anomaly['regulatory_narrative']}")
            
            if 'top_factors' in first_anomaly and first_anomaly['top_factors']:
                print(f"\nTop Contributing Factors:")
                for i, factor in enumerate(first_anomaly['top_factors'][:3], 1):
                    print(f"  {i}. {factor['feature']}: {factor['value']:.4f} "
                          f"(peer avg: {factor['peer_mean']:.4f}, "
                          f"z-score: {factor['z_score']:.2f})")
                    print(f"     Reason: {factor['reason']}")
        elif 'total_anomalies' in anom_section and anom_section['total_anomalies'] == 0:
            print("✓ No anomalies detected in this audit (system functioning correctly)")
        else:
            print("✗ No anomaly explanations found")
    else:
        print("✗ No anomaly detection results")
    
    # Check credit risk explanations
    print("\n2. CREDIT RISK EXPLANATIONS")
    print("-" * 80)
    
    if 'risk_assessments' in results and 'credit_risk' in results['risk_assessments']:
        credit_section = results['risk_assessments']['credit_risk']
        
        if 'regulatory_narrative' in credit_section and credit_section['regulatory_narrative']:
            print("✓ Credit risk explanation exists")
            print(f"\nRegulatory Narrative:")
            print(f"  {credit_section['regulatory_narrative']}")
        
        if 'ml_explanation' in credit_section:
            explanation = credit_section['ml_explanation']
            
            if 'model_insights' in explanation and explanation['model_insights']:
                print(f"\nML Model Insights:")
                for model_name, insight in explanation['model_insights'].items():
                    verdict = insight.get('verdict', 'N/A')
                    print(f"  • {model_name}: {verdict}")
            
            if 'key_factors' in explanation and explanation['key_factors']:
                print(f"\nKey Risk Factors:")
                for i, factor in enumerate(explanation['key_factors'][:3], 1):
                    print(f"  {i}. {factor['metric']}: {factor['value']:.4f} "
                          f"(peer avg: {factor['peer_mean']:.4f}, "
                          f"deviation: {factor['pct_deviation']:.1f}%)")
                    print(f"     {factor['reason']}")
            else:
                print("\n(No significant metric deviations detected)")
        else:
            print("✗ No ML explanation found in credit risk results")
    else:
        print("✗ No credit risk results")
    
    # Check liquidity risk explanations
    print("\n3. LIQUIDITY RISK EXPLANATIONS")
    print("-" * 80)
    
    if 'risk_assessments' in results and 'liquidity_risk' in results['risk_assessments']:
        liquidity_section = results['risk_assessments']['liquidity_risk']
        
        if 'regulatory_narrative' in liquidity_section and liquidity_section['regulatory_narrative']:
            print("✓ Liquidity risk explanation exists")
            print(f"\nRegulatory Narrative:")
            print(f"  {liquidity_section['regulatory_narrative']}")
        
        if 'ml_explanation' in liquidity_section:
            explanation = liquidity_section['ml_explanation']
            
            if 'stress_indicators' in explanation and explanation['stress_indicators']:
                print(f"\nStress Test Indicators:")
                for key, value in explanation['stress_indicators'].items():
                    print(f"  • {key}: {value}")
            
            if 'key_factors' in explanation and explanation['key_factors']:
                print(f"\nKey Risk Factors:")
                for i, factor in enumerate(explanation['key_factors'][:3], 1):
                    print(f"  {i}. {factor['metric']}: {factor['value']:.4f} "
                          f"(peer avg: {factor['peer_mean']:.4f}, "
                          f"deviation: {factor['pct_deviation']:.1f}%)")
                    print(f"     {factor['reason']}")
            else:
                print("\n(No significant metric deviations detected)")
        else:
            print("✗ No ML explanation found in liquidity risk results")
    else:
        print("✗ No liquidity risk results")
    
    # Check comprehensive report
    print("\n4. COMPREHENSIVE REPORT")
    print("-" * 80)
    
    if 'risk_assessments' in results:
        risk_assessments = results['risk_assessments']
        
        # Check credit risk section
        if 'credit_risk' in risk_assessments:
            cr_section = risk_assessments['credit_risk']
            if 'regulatory_narrative' in cr_section and cr_section['regulatory_narrative']:
                print("✓ Credit risk narrative included in report")
            if 'ml_explanation' in cr_section and cr_section['ml_explanation']:
                print("✓ Credit risk ML explanation included in report")
        
        # Check liquidity risk section
        if 'liquidity_risk' in risk_assessments:
            lr_section = risk_assessments['liquidity_risk']
            if 'regulatory_narrative' in lr_section and lr_section['regulatory_narrative']:
                print("✓ Liquidity risk narrative included in report")
            if 'ml_explanation' in lr_section and lr_section['ml_explanation']:
                print("✓ Liquidity risk ML explanation included in report")
        
        # Check anomaly detection section
        if 'anomaly_detection' in risk_assessments:
            anom_section = risk_assessments['anomaly_detection']
            if 'explained_anomalies' in anom_section:
                print(f"✓ Anomaly explanations structure included in report")
    else:
        print("✗ No risk assessments in report")
    
    # Save detailed results for inspection
    print("\n5. SAVING DETAILED RESULTS")
    print("-" * 80)
    
    output_file = Path("test_transparency_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert to JSON-serializable format
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Detailed results saved to {output_file}")
    
    print("\n" + "="*80)
    print("REGULATORY TRANSPARENCY TEST COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    test_regulatory_explanations()
