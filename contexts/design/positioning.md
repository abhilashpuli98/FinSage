# FinSage Positioning

## Quick Summary

- **Purpose**: Describe FinSage's current research scope, implemented foundations, planned architecture, and upstream attribution.
- **Read when**: Reviewing project descriptions, deciding whether a capability is shipped, or adding a research direction.
- **Load next**: Read the root [README](../../README.md) for user-facing installation, examples, and implementation status.
- **Authority**: Current project framing; code and tests remain the authority for runtime behavior.

## Contents

- [Project Identity](#project-identity)
- [Current Capabilities](#current-capabilities)
- [Research Questions](#research-questions)
- [Target Architecture](#target-architecture)
- [Upstream Attribution](#upstream-attribution)

## Project Identity

FinSage is open-source research software for financial knowledge extraction and retrieval. The distribution and Python import namespace are `finsage`. This checkout preserves the original QuantMind code lineage and licensing while developing temporal and conflict-aware research capabilities; that history does not make upstream publications FinSage results.

## Current Capabilities

The repository currently includes typed flat and tree knowledge, paper and news processing operations, page-aware PDF parsing, deterministic BM25 retrieval, SQLite-backed local knowledge storage, temporal knowledge models and paper extraction, conflict detection and resolution components, LiteLLM model routing, agentic retrieval over document structure trees, and experimental evaluation utilities.

Persistent user memory, a concrete knowledge graph, an integrated multi-agent research workflow, cross-encoder reranking, and runtime claim verification are not implemented as an end-to-end system. See the root README's implementation table for current boundaries.

## Research Questions

FinSage investigates whether explicit validity windows, conflict-aware evidence handling, and configurable local/API model routing can improve the reliability and reproducibility of financial knowledge workflows. These are hypotheses to evaluate against suitable baselines; no performance improvement or benchmark result is claimed without completed experiments.

## Target Architecture

The root `architecture.png` illustrates the longer-term two-stage design: knowledge extraction and engineering, followed by retrieval and reasoning. Its knowledge graph, separate vector and graph databases, persistent memory, general multi-agent orchestration, reranking, and answer-verification components are planned integration points unless the implementation status documents a narrower shipped component.

## Upstream Attribution

FinSage derives from the [LLMQuant/quant-mind repository](https://github.com/LLMQuant/quant-mind). The original project research citation is [arXiv:2509.21507](https://arxiv.org/abs/2509.21507). Preserve the upstream authorship and existing copyright notices; do not describe the original paper as a FinSage publication or claim its results for this project.
