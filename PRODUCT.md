# PRODUCT.md

## Product Purpose

`graph_invariant` is a research tool for discovering useful graph invariants with LLM-assisted search.  
It aims to help researchers generate, evaluate, and compare candidate formulas against known structural graph metrics.

## Target Users

- Graph/network science researchers
- Applied ML researchers exploring symbolic discovery
- Engineers prototyping interpretable graph descriptors

## Core User Problems

- Finding high-signal graph descriptors is slow and manual.
- Candidate formulas are hard to compare consistently across datasets.
- Reproducibility is difficult without standardized artifacts and checkpoints.

## Key Features

- Deterministic graph dataset generation for train/validation/test splits
- LLM-driven candidate generation with island-style exploration
- Sandboxed candidate execution with timeout and memory constraints
- Composite scoring:
  - predictive quality
  - simplicity
  - novelty vs known invariants
- Baseline comparisons (statistical + optional PySR)
- JSON/JSONL artifacts, checkpoints, and markdown report generation
- Multi-seed benchmark mode

## Product Objectives

- Short term:
  - Provide a reliable Phase 1 research loop with reproducible outputs.
  - Make experiment outcomes auditable through structured artifacts.
- Medium term:
  - Improve discovery quality under fixed compute budgets.
  - Improve reliability of novelty and baseline comparisons.
- Long term:
  - Support publication-quality experiments and OSS reuse.

## Non-Goals (Current Scope)

- Production-grade sandboxing for arbitrary third-party code
- Distributed cluster orchestration
- Web UI and managed experiment tracking platform

## Success Criteria

- Pipeline completes end-to-end from config to report.
- CI stays green with deterministic tests.
- Artifacts are sufficient to reproduce and review experiment outcomes.
