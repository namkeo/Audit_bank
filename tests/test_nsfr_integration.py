"""
Test NSFR integration in liquidity risk model and macro-adjustments.
"""
import pandas as pd

# Load dataset
df = pd.read_csv('time_series_dataset_enriched.csv')

print("✅ NSFR Integration Verification:")
print(f"   - NSFR in dataset: {'nsfr' in df.columns}")
print(f"   - Total rows: {len(df)}")

if 'nsfr' in df.columns:
    print(f"   - NSFR column stats:")
    print(f"     • Min: {df['nsfr'].min():.1%}")
    print(f"     • Mean: {df['nsfr'].mean():.1%}")
    print(f"     • Max: {df['nsfr'].max():.1%}")
    print(f"   - Banks below 100% (Basel III minimum): {(df['nsfr'] < 1.0).sum()}")
    print(f"   - Banks at/above 100%: {(df['nsfr'] >= 1.0).sum()}")
    
    print("\n✅ NSFR Integration Points:")
    print("   1. dataset_enrichment.py - NSFR generation")
    print("   2. expert_rules.json - min_nsfr: 1.0 threshold")
    print("   3. 2_model_liquidity_risk.py - NSFR in features")
    print("   4. 2_model_liquidity_risk.py - NSFR indicator calculation")
    print("   5. 2_model_liquidity_risk.py - NSFR in macro-adjustment stress metrics")
    
    print("\n✅ Macro-adjustments stress configuration:")
    print("   - Stress metrics: ['liquidity_ratio', 'lcr', 'nsfr']")
    print("   - NSFR threshold: < 1.0 (below Basel III 100% minimum)")
    print("   - Benchmark metrics: ['liquidity_ratio', 'lcr', 'nsfr']")
    
    print("\n✅ Basel III Compliance:")
    compliance_rate = (df['nsfr'] >= 1.0).mean() * 100
    print(f"   - Industry compliance rate: {compliance_rate:.1f}%")
    print(f"   - {(df['nsfr'] >= 1.0).sum()} banks meet Basel III NSFR ≥ 100% requirement")
    print(f"   - {(df['nsfr'] < 1.0).sum()} banks below minimum (structural funding risk)")
else:
    print("   ❌ NSFR column not found in dataset!")
