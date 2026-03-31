# Radiology Pre-Screening Subnet (TB First)

## Bittensor Subnet Ideathon Proposal

### Summary

We propose a Bittensor subnet focused on **radiology pre-screening AI** for **multiple high-value chest conditions** (currently including **Tuberculosis, Pneumonia, Bronchitis, and Silicosis**). The subnet is designed to drive the development of **state-of-the-art (SOTA)** models for this domain through a measurable, incentive-driven competition with **continuous evaluation over time-bucketed periods**.

The core thesis is simple: this is a high-value, high-demand domain with clear evaluation targets, continuous data generation, and strong need for generalization. These properties make it well-suited for Bittensor’s miner-validator incentive model.

---

## Why This Subnet Should Exist

Radiology systems in many regions face a structural gap between imaging volume and radiologist capacity. This creates delays in review and prioritization. Pre-screening AI can help triage and prioritize cases for faster human review.

This is also a commercially attractive domain:

* multiple AI companies are actively approaching medical data archiving companies to obtain training data for radiology AI,
* which signals strong market demand and urgency,
* but most of these efforts are centralized and closed.

Our subnet aims to prove that a decentralized incentive network can compete with — and potentially outperform — centralized AI vendors by accelerating model improvement through open competition.

---

## Core Subnet Objective

Build a Bittensor subnet that produces **SOTA radiology pre-screening models** by rewarding miners for model performance on data generated **after** their on-chain model commitment.

This is not a static benchmark subnet. It is a **continuous model improvement system**, where evaluation happens in **fixed-length time periods** (e.g., every 10 minutes or every 24 hours) and historical performance across recent periods determines rewards.

---

## Incentive Mechanism (Core Design)

### What Miners Produce

Miners do **not** merely return predictions as the primary output.
The miner’s output is a **trained model** that can be queried via an OpenAI-compatible API.

Miner workflow (current implementation):

1. Train a radiology pre-screening model.
2. Publish/version the model artifact on **Hugging Face** (repo + revision).
3. Preferably host the model on **Chutes (SN64)** and expose it via a `chute_id`, or
4. Optionally, allow validators to run the HF snapshot **locally** via SGLang (when `--allow-local` is enabled).
5. Submit a **commitment on-chain** with JSON metadata:
   * `repo`: Hugging Face repo id,
   * `revision`: Git revision / tag,
   * `chute_id`: Chutes deployment id (may be empty when local hosting is allowed).

### What Validators Do

Validators evaluate miners’ models (Chutes-hosted or locally hosted via SGLang) using standardized requests and scoring logic.

The key requirement is **temporal evaluation integrity**:

* validators evaluate each submitted model on data that the model **could not have used during its training period**, approximated by an on-chain commit block and a configurable **evaluation delay**,
* and rewards are allocated based on **measured performance aggregated over recent evaluation periods**.

### Reward Mechanism: Tiered Emission with Strong Top Pressure

Instead of a literal single-winner-takes-all, the subnet uses a **tiered emission mechanism** that heavily rewards the top performer while still giving secondary incentives to other strong models:

At each weight-setting cycle:

* Validators maintain a SQLite database of **daily (or per-period) scores** for each UID and evaluation period.
* For each UID, the validator aggregates Fβ performance over a recent window of periods.
* Several **tiers (A–E)** are defined, each with:
  * a **lookback window in evaluation periods** (e.g., last 5, 4, 3, 2, 1 periods),
  * a rule for selecting top miners (top-1 or top percentage),
  * and an **emission share** (e.g., Tier A = 95% of raw emission).
* Within each tier, miners are ranked by:
  * mean Fβ over the tier window,
  * then mean recall,
  * then mean precision,
  * then **smaller model size** (fewer parameters from Hugging Face) as a tie-breaker,
  * then earlier on-chain commit block.
* Tier emissions are **additive** across tiers, and the resulting per-UID scores are normalised to sum to 1.0 before calling `set_weights`.

In practice, this behaves like a **soft winner-takes-most** system:

* Tier A’s single top miner receives the majority (e.g., 95%) of raw emissions,
* lower tiers distribute the remaining emissions to a broader set of strong miners,
* but low-performing or unevaluated miners receive effectively zero.

This preserves **strong competitive pressure for SOTA performance** while:

* stabilising incentives across recent periods,
* making the reward signal less brittle to one-off fluctuations,
* and allowing high-performing newcomers to appear in lower tiers before reaching Tier A.

### Time-Shifted Evaluation (Important Clarification)

The evaluation data is **not required to be secret or hidden**.

The key mechanism is **time delay**, not secrecy:

* if a model is committed on-chain at block/time **T**,
* it is only evaluated on samples whose acquisition timestamps are **strictly after** `T + eval_delay`,
* where `eval_delay` is a configurable delay window (currently defaulting to approximately one day in production, and set to zero for mock/test modes).

Miners can still use previously evaluated data for future training.
That is acceptable and expected.

What matters is:

* a model is only scored on data that did not exist (or was not yet available) during that model’s training period as approximated by its chain commit time.

This rewards **generalization**, not static benchmark tuning.

---

## Evaluation Delay Window (Implemented and Tunable)

The subnet exposes an **evaluation delay window** as a validator parameter:

* `--eval-delay-minutes` / `EVAL_DELAY_MINUTES`,
* defaulting to a value that approximates a **1-day minimum delay** between on-chain commit and first eligible evaluation in production,
* and set to 0 in mock / local test modes to allow fast iteration.

This window can be tuned over time based on:

* data availability cadence,
* operational practicality,
* and how effectively the delay prevents short-cycle overfitting while preserving rapid iteration.

The core principle remains: models are evaluated on **future-period data** relative to their on-chain commitment.

---

## Why Bittensor Fits This Use Case

This domain is a strong fit for Bittensor because it has the key properties required for a high-quality incentive network:

* **Measurable performance** (e.g., sensitivity, specificity, false-negative behavior, calibration)
* **Continuous data generation** (new imaging and reports over time)
* **Strong value of generalization** (future-case performance matters more than benchmark performance)
* **High-value market demand** (multiple companies already competing for access to data in this domain)

Bittensor provides the mechanism to coordinate many independent teams and reward only those that measurably improve model quality.

---

## Why This Can Beat Centralized AI Vendors

Centralized AI vendors typically rely on a single internal research pipeline. In contrast, this subnet creates:

* parallel model development from many independent teams,
* transparent competition on measurable outcomes,
* continuous retraining and iteration,
* and open model publication / auditability.

This structure can increase the rate of improvement and make model performance more contestable and evidence-driven.

---

## Data and Partnership Model (High-Level)

The subnet relies on partnerships with medical data archiving companies for redacted/de-identified imaging and associated reports/metadata (as permitted).

This is foundational because:

* data quality and continuity determine subnet quality,
* and ongoing data flow enables continuous evaluation and improvement.

The subnet begins with radiology pre-screening for TB and is designed to expand to additional disease categories over time.

---

## Scope and Positioning

This subnet is for **pre-screening / triage support**, not diagnosis replacement.
Human clinicians (radiologists) remain the final decision-makers.

---

## Closing Statement

This subnet is designed to build **SOTA radiology pre-screening models** through a clear, technically grounded incentive mechanism:

* miners produce trained models and register them on-chain via HF repo + revision (and optionally Chutes deployment),
* validators evaluate those models via a standard OpenAI-compatible interface on future-period data the models could not have used at training time (within a configurable evaluation delay),
* and a **tiered, winner-takes-most** reward structure directs the majority of incentives to the best-performing miner while still recognising other strong contributors over recent evaluation periods.

We believe this is one of the strongest real-world applications of Bittensor’s incentive design: a high-demand domain, clear utility, continuous data, and a credible path to outperform centralized AI development through decentralised competition and open, period-based evaluation.
