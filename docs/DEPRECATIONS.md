# Deprecations

This document tracks deprecated APIs and compatibility shims removed or scheduled for removal as part of streamlining the object-oriented design.

## Policy
- Deprecated methods emit a `DeprecationWarning` and a log warning via the centralized logger.
- Where backward compatibility is necessary, prefer introducing a small `Legacy*` subclass that preserves the old interface instead of cluttering core classes.
- Redundant proxy methods that only forward to newer time-series APIs will be removed unless there is demonstrated usage.

## Changes (2026-01-08)
- Removed `DataPreparation._calculate_ratios_for_period(...)` which only delegated to `FinancialRatioCalculator.calculate_ratios(...)`.
  - Rationale: No usage across the workspace; modern API is `calculate_time_series_ratios()`.
  - Alternative: Use `FinancialRatioCalculator.calculate_ratios(series, data_format='series')` directly or `DataPreparation.calculate_time_series_ratios()` for batch processing.

## Deprecation Utilities
- A `deprecated(...)` decorator was added to `6_logging_config.py` to mark methods as deprecated while providing warnings and logging.

