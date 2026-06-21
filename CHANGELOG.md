# Changelog

All notable changes to the Digital Spotter project will be documented in this file.

## [Unreleased]

### Added
- PHASE_0: Project bootstrap — directory structure, dependencies, git setup

### Changed
- Upgraded inference pipeline to support v4 BiLSTM+MHA model
- Expanded feature vector from 38 to 40 features
- Added 4th synthetic timestep (delta sequence) for temporal context
- Replaced scaler_wrapper with preprocessor for full joblib artifact chain (winsorize, impute, scale)
- Updated session metadata and database schema to track subject height, weight, and FTR
