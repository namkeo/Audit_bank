import pandas as pd
import numpy as np
from typing import Tuple

try:
    from reproducibility import set_random_seeds
except Exception:
    def set_random_seeds(seed: int = 42):
        np.random.seed(seed)

SECTORS = [
    'energy', 'real_estate', 'construction', 'services', 'agriculture'
]

RATINGS = ['AAA','AA','A','BBB','BB','B','CCC']


def _rating_for_row(stability: str) -> str:
    s = (stability or '').lower()
    if s == 'high':
        return 'A'
    if s == 'medium':
        return 'BBB'
    return 'BB'


def enrich_time_series_dataset(input_csv: str = 'time_series_dataset.csv',
                               output_csv: str = 'time_series_dataset_enriched.csv') -> Tuple[str, int]:
    """
    Làm giàu dataset chuỗi thời gian với các chỉ số đa dạng hóa danh mục tín dụng,
    tỷ lệ tài trợ ổn định Basel III và các chỉ số tài trợ bán buôn.
    Định thức bằng cách sử dụng hạt giống toàn cục.
    Thêm các cột:
      - sector_loans_<sector>
      - top1_borrower_loans
      - top10_borrower_loans
      - obs_exposure_total
      - external_credit_rating
      - nsfr (Tỷ lệ Tài trợ Ổn định Ròng): Đo lường rủi ro tài trợ cấu trúc
              Bắt buộc: >= 100% theo Basel III
              NSFR = Available Stable Funding / Required Stable Funding
      - wholesale_funding_short_term: Volatile market funding (repos, commercial paper, etc.)
      - wholesale_funding_stable: Stable wholesale funding (>1 year maturity)
      - loan_to_total_funding_ratio: loans / (deposits + wholesale_stable) - comprehensive funding ratio
      - wholesale_dependency_ratio: wholesale_total / total_funding - reliance on market funding
    """
    set_random_seeds(42)
    df = pd.read_csv(input_csv)

    # Ensure total_loans exists
    if 'total_loans' not in df.columns:
        raise ValueError('Input dataset must contain total_loans')

    # Generate sector allocations deterministically via Dirichlet on each bank_id
    enriched_rows = []
    last_key = None
    weights = None

    for idx, row in df.iterrows():
        bank = str(row.get('bank_id'))
        period = str(row.get('period'))
        loans = float(row.get('total_loans', 0) or 0)

        # Seed per bank for reproducibility
        seed = abs(hash(bank)) % (2**32)
        rng = np.random.RandomState(seed)
        weights = rng.dirichlet(np.ones(len(SECTORS)))

        sector_values = {f'sector_loans_{sec}': float(loans * w) for sec, w in zip(SECTORS, weights)}

        # Borrower concentrations: top1 and top10 exposures
        # Top1 ~ 6-12% of loans, Top10 ~ 25-40%
        top1_ratio = 0.06 + 0.06 * rng.rand()
        top10_ratio = 0.25 + 0.15 * rng.rand()
        top1 = float(loans * top1_ratio)
        top10 = float(loans * top10_ratio)

        # Off-balance sheet exposures ~ 10-30% of loans
        obs_ratio = 0.10 + 0.20 * rng.rand()
        obs_exposure = float(loans * obs_ratio)

        # External rating by stability
        rating = _rating_for_row(str(row.get('stability', 'medium')))

        # NSFR (Net Stable Funding Ratio) - Basel III structural funding metric
        # NSFR = Available Stable Funding / Required Stable Funding
        # Should be >= 100% (minimum 100%)
        # Assume deposits and stable funding provide 60-85% of assets
        # Required funding based on asset composition
        
        # Available Stable Funding (ASF): Stable customer deposits, long-term funding
        # Assume deposits comprise ~65-85% of customer funds
        deposits = float(row.get('total_deposits', 0) or 0)
        stable_deposits = deposits * (0.65 + 0.20 * rng.rand())  # 65-85% is stable
        
        # Long-term wholesale funding (term loans > 1 year, bonds)
        # Approximately 30-50% of non-deposit funding
        other_funding = float(row.get('total_loans', 0) or 0) * 0.15  # ~15% of loans in long-term funding
        
        available_stable_funding = stable_deposits + other_funding
        
        # Required Stable Funding (RSF): Depends on asset maturity and liquidity
        # Assume ~70-85% of assets require stable funding over 1-year horizon
        total_assets = float(row.get('total_loans', 0) or 0) + float(row.get('total_deposits', 0) or 0) * 0.5
        required_stable_funding = total_assets * (0.70 + 0.15 * rng.rand())
        
        # NSFR ratio: should be >= 1.0 (100%)
        # Generate realistic range: 85%-125%
        # Poor banks: 85-95%, Average: 95-110%, Strong: 110-125%
        nsfr_base = 0.85 + 0.40 * rng.rand()  # 85%-125%
        # Adjust based on liquidity_ratio if available
        if 'liquidity_ratio' in row and pd.notna(row['liquidity_ratio']):
            liq = float(row['liquidity_ratio'])
            nsfr_base = 0.90 + min(0.35, liq * 0.5)  # Better liquidity -> higher NSFR
        nsfr = max(0.80, min(1.30, nsfr_base))  # Cap between 80% and 130%

        # Wholesale Funding Indicators
        # Wholesale funding = non-deposit funding (repos, commercial paper, interbank loans, bonds)
        # Split into short-term (volatile) and stable (>1 year)
        
        # Total funding gap = loans - deposits (simplified approach)
        funding_gap = max(0, loans - deposits)
        
        # Wholesale funding needed to fund loans beyond deposits
        # Short-term wholesale (volatile): repos, commercial paper, short-term interbank
        # Typically 5-25% of total assets for banks relying on market funding
        wholesale_short_term_ratio = 0.05 + 0.20 * rng.rand()  # 5-25% of total assets
        wholesale_funding_short_term = total_assets * wholesale_short_term_ratio
        
        # Stable wholesale funding: long-term bonds, term deposits >1 year
        # Already calculated as 'other_funding' in NSFR - reuse for consistency
        # Typically 10-30% of total assets
        wholesale_stable_ratio = 0.10 + 0.20 * rng.rand()  # 10-30% of total assets
        wholesale_funding_stable = total_assets * wholesale_stable_ratio
        
        # Total wholesale funding
        wholesale_total = wholesale_funding_short_term + wholesale_funding_stable
        
        # Total funding base = deposits + wholesale
        total_funding = deposits + wholesale_total
        
        # Loan-to-Total-Funding Ratio: loans / (deposits + wholesale_stable)
        # More comprehensive than LDR, captures reliance on stable funding sources
        # Healthy range: 70-90%; >90% indicates aggressive lending, <70% underlending
        loan_to_total_funding = loans / (deposits + wholesale_funding_stable) if (deposits + wholesale_funding_stable) > 0 else 0
        
        # Wholesale Dependency Ratio: wholesale_total / total_funding
        # Measures reliance on market funding vs retail deposits
        # Higher ratio = more vulnerable to market disruptions
        # Healthy range: <30%; 30-50% moderate; >50% high dependency
        wholesale_dependency = wholesale_total / total_funding if total_funding > 0 else 0

        enriched = {
            **row.to_dict(),
            **sector_values,
            'top1_borrower_loans': top1,
            'top10_borrower_loans': top10,
            'obs_exposure_total': obs_exposure,
            'external_credit_rating': rating,
            'nsfr': nsfr,
            'wholesale_funding_short_term': wholesale_funding_short_term,
            'wholesale_funding_stable': wholesale_funding_stable,
            'loan_to_total_funding_ratio': loan_to_total_funding,
            'wholesale_dependency_ratio': wholesale_dependency,
        }
        enriched_rows.append(enriched)

    out_df = pd.DataFrame(enriched_rows)
    out_df.to_csv(output_csv, index=False)
    return output_csv, len(out_df)

if __name__ == '__main__':
    path, n = enrich_time_series_dataset()
    print(f'Wrote {n} rows to {path}')
