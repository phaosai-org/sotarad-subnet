# Technical architecture: miner and validator workflows

This document describes **operational logic** for the radiology pre-screening subnet (see `docs/PROPOSAL.md`): workflows, data rules, configuration boundaries, and the **evaluation metric** (including one reference figure for **Fβ**). Training pipelines and miner client code are **out of scope** for this repository; the subnet defines how commitments are interpreted and how validators score models.

---

## 1. Scope and design axis

**What is measured:** each miner’s model is scored with the **Fβ score** (see §4.4) on evaluation windows that are **strictly after** that model’s on-chain submission time, using a **public** evaluation dataset API.

**Why not plain accuracy or F1:** pre-screening must **minimize missed disease** (high **recall / sensitivity**). Operational efficiency and human workload also depend on **precision** (fewer false alarms). **Recall is more important than precision**, but precision still matters. **F1** weights precision and recall equally; **Fβ** with **β greater than 1** increases the penalty on low recall relative to low precision, which matches “don’t miss positives” better than F1.

- Training and fine-tuning happen off-chain and are **not** part of this repo.
- Miners expose inference via **Chutes**; validators call the public inference API.
- Model identity and deployment pointers are published **on-chain as commits**; commit block/time is the **model submission timestamp**.

---

## 2. On-chain commitment (miners)

### 2.1 What miners commit

Each miner submits a chain commitment that encodes (at minimum) everything validators need to locate and invoke the model:

- Hugging Face model identifier (repo / revision or equivalent versioning fields as specified by subnet schema).
- Chutes deployment identifier (or endpoint metadata required for standardized inference calls).
- Any additional fields required for duplicate detection and ranking tie-breaks (e.g., canonical model size metric—see §6).

The exact commitment encoding (JSON in commit body, hashing, size limits) is an implementation detail; architecturally, **the commitment is the source of truth** for “which model is registered for this hotkey/UID at this time.”

### 2.2 Submission time

**Submission timestamp** for eligibility and evaluation windows is derived from the **on-chain commit** (e.g., block timestamp or agreed subnet convention), not from Hugging Face or Chutes upload time.

### 2.3 First evaluation day

If a model is committed on calendar day **D** (per submission timestamp), validators treat it as **not yet in the daily evaluation set for D**. **First eligible evaluation day is D+1** (next calendar day under UTC or a single fixed timezone defined by implementation). All daily logic in §4 uses this rule consistently.

### 2.4 Reference pattern: Affine (HF revision + Chutes + chain commit)

This subnet’s **miner submission shape** is intentionally parallel to **Affine**, which already runs the “Hugging Face artifact + Chutes deployment + Bittensor commitment” loop in production. See the Affine Cortex repo and miner guide for operational detail: [AffineFoundation/affine-cortex](https://github.com/AffineFoundation/affine-cortex), [docs/MINER.md](https://github.com/AffineFoundation/affine-cortex/blob/main/docs/MINER.md).

**What Affine documents (abridged, for alignment):**

- **Accounts:** Hugging Face (host weights), [Chutes](https://github.com/chutesai/chutes) (host inference; Affine recommends registering Chutes with the **same hotkey** as the Bittensor mining hotkey), Bittensor wallet registered on the target subnet.
- **Version pin:** Upload the model to Hugging Face and record the **git revision SHA** of that artifact (not only a branch name)—Affine treats `--revision <SHA>` as the canonical version for deploy and commit.
- **Deploy:** Push the pinned HF repo+revision to Chutes; the deploy step yields a `**chute_id`** (deployment id) validators use to route requests.
- **On-chain:** Commit a small payload that binds **HF repo id**, **revision SHA**, and `**chute_id`** so validators can resolve inference for each UID (Affine’s CLI: `af commit --repo <user/repo> --revision <SHA> --chute-id <id>`).

**How this subnet uses the same idea:**

- Validators read the **latest valid commitment** per miner and parse at least `**repo` + `revision` + `chute_id`**, plus any **extra fields** we require (e.g., declared parameter count for §6 tie-breaks). Encoding (JSON, size limits, hashing) is implementation-defined but should stay **compact** because chain commitment updates are **rate-limited** on Bittensor (order of **one meaningful update per ~100 blocks** is the usual pattern—store full pointers once per model version, not per file).

**Difference from Affine’s validator:** Affine validators often **fetch weights from a central API** and set weights on-chain ([docs/VALIDATOR.md](https://github.com/AffineFoundation/affine-cortex/blob/main/docs/VALIDATOR.md)). This subnet’s validators **run daily evaluation locally** (dataset API + Chutes calls) and derive weights from tier rules in §5. The **miner/on-chain payload pattern** still matches Affine’s proven workflow.

---

## 3. Miner workflow (end-to-end)

1. Train or refine a model **outside** this repo (private or public pipeline—subnet-agnostic).
2. Upload the trained artifact to **Hugging Face** and pin a **revision SHA** (see §2.4).
3. Deploy that **repo + revision** to **Chutes**; obtain `**chute_id`** (or equivalent deployment handle).
4. Submit an **on-chain commit** binding HF **repo**, **revision**, **chute_id**, and any required extra fields (§2.1). This registers the model and fixes **submission time** for temporal evaluation rules.
5. Update the commitment only when publishing a new model version; each update is subject to chain rate limits—miners should **not** commit per small change. Validators use the **current** commitment for daily evaluation.

Miners do **not** send training data or weights through the validator in this design; validators **pull evaluation data** from the configured dataset service and **push inference requests** to Chutes.

**Operational reference:** Step-by-step env vars, `chutes register`, funding, and CLI examples are maintained upstream in Affine’s miner guide—not duplicated here. Use it as a template for Chutes + HF + wallet setup; substitute this subnet’s **netuid** and **commit schema** when implemented.

---

## 4. Validator workflow (daily, strict)

Validation runs on a **strict daily cadence** (one evaluation pass per calendar day per configured schedule). “Configurable” items are **tier tables, delays, dataset base URL, and query parameters**—not the choice to run ad-hoc within a day unless explicitly extended later.

### 4.1 Discover registrations

1. Sync metagraph / chain state for the subnet.
2. For each active miner UID, read the **latest valid on-chain commitment** and parse model metadata (**HF repo**, **revision**, **`chute_id`**, plus any subnet-specific fields—same logical fields Affine exposes via queries like `af get-miner <UID>` in [affine-cortex](https://github.com/AffineFoundation/affine-cortex)).

### 4.2 Duplicate detection

Validators determine which committed models are **duplicates** of each other (same or trivially equivalent artifacts across UIDs). Exact **duplicate** criteria are implementation-defined (e.g., HF repo + revision equality, hash of weights if exposed, Chutes deployment mapping).

**Eligibility rule (fixed policy):** for each set of UIDs that share the same duplicate identity, **only the UID that submitted that model earliest on-chain remains eligible** for evaluation and incentives. **Submission order** is defined by **on-chain commit timestamp** (same convention as §2.2). **All later duplicate submitters are not eligible**: they are excluded from daily scoring and from tier / weight allocation for that duplicate artifact until they commit a **distinct** (non-duplicate) model. This avoids punishing the original submitter and removes incentive to re-register identical weights under a new hotkey.

If two commits share the same timestamp (e.g. same block), use a **deterministic tie-break** (e.g. lower numeric UID, or lexicographic hotkey)—documented once in implementation.

Architecture requirement: **validators perform duplicate grouping** using on-chain metadata and public artifact references, then **mark ineligible** every duplicate UID except the earliest submitter in each group, **before** running inference (§4.4).

### 4.3 Evaluation data (public API, time-shifted)

- Validators use a **configurable public dataset base URL** (single logical service; may include path prefix).
- For each evaluation day **E**, validators request samples for a **time range** that satisfies:
  - Data is **fresh for that daily run** (fetched for the evaluation window being scored, not a static local cache from training era).
  - For a model with submission time **T_sub**, the evaluation interval must be **strictly after** **T_sub** plus a **configurable evaluation delay** (cutoff / embargo), so that the data could not have been part of a good-faith training run submitted at **T_sub** under subnet rules.

Concretely: implementation fixes a rule such as “only use samples with `timestamp > T_sub + delay`” or “only use the API’s window `[start, end]` that lies entirely after that cutoff,” where **`delay` is validator-configurable** (proposal suggests on the order of 1–7 days; exact default is not fixed here).

- The **query shape** (query parameters for time period, pagination, task filters) is part of validator configuration so the same client can target different dataset deployments.

### 4.4 Model evaluation: precision, recall, and Fβ

#### 4.4.1 Inference pass

For each **eligible** UID on evaluation day **E** (respecting §2.3 and §4.3), validators send standardized requests to the miner’s **Chutes** endpoint for every sample in that day’s evaluation batch (labels come from the dataset API / ground-truth provider, not from the miner).

#### 4.4.2 Task definition and confusion matrix

Evaluation is treated as **binary** at the scoring layer: a **positive** label means “screen positive / flag for further work” (disease-present or equivalent subnet-defined positive class); **negative** is the complement. From aggregate counts over the batch:

- **True positives (TP):** predicted positive, actually positive.  
- **False negatives (FN):** predicted negative, actually positive (**missed case**—especially harmful in screening).  
- **False positives (FP):** predicted positive, actually negative.  
- **True negatives (TN):** predicted negative, actually negative.

Then, over the scored batch:

- **Precision** is the fraction of predicted positives that are correct: **TP / (TP + FP)** (of all cases the model called positive, how many were truly positive).
- **Recall** (same as **sensitivity**) is the fraction of actual positives the model caught: **TP / (TP + FN)** (of all true disease-positive cases, how many the model flagged).

Implementation must define behavior when denominators are zero (e.g. **TP + FP = 0** or **TP + FN = 0**)—skip the day, assign a sentinel score, or use a documented fallback—so ranking stays well-defined.

#### 4.4.3 Fβ formula (primary score)

The **primary scalar score** for ranking and tiers is **Fβ**, where **β** (beta) is a **validator-configurable** positive number.

**In words:** Fβ blends precision and recall into one number. The parameter **β** controls how much **recall** is weighted **relative to precision**: larger **β** means recall matters more in the blend; **β = 1** is the familiar **F1** (equal weight); **β less than 1** tilts toward precision.

**As a formula** (same as common libraries such as [scikit-learn `fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)`):

```text
Fβ = (1 + β²) × (Precision × Recall) / ((β² × Precision) + Recall)
```

Equivalently: take the product **Precision × Recall**, multiply by **(1 + β²)**, and divide by **(β² × Precision + Recall)**.

- **β = 1:** this is **F1** (harmonic mean of precision and recall).
- **β greater than 1:** emphasizes **recall**—appropriate when **false negatives** (missed disease) hurt more than **false positives**; **F2** (β = 2) is a common screening-oriented choice.
- **β less than 1:** emphasizes **precision**—not the default intent for this subnet.

**Interpretation:** in the usual reading, **β** reflects **how many times more important recall is than precision** when trading them off inside this single score (see standard ML references and the figure below).

**Subnet default:** expose **β** in validator config (e.g. default **β = 2** if the subnet wants recall to dominate the blend in line with F2-style screening—tune empirically).

**Implementation note:** match one documented convention and unit-test edge cases (empty batches, zero denominators).

#### 4.4.4 How varying β changes the score (reference figure)

The following figure fixes **precision at 0.5** and shows how **Fβ** rises as **recall** increases, for three settings: **F0.5** (precision-heavy), **F1** (balanced), and **F2** (recall-heavy). For the same recall, a **larger β** gives a **steeper** reward for improving recall—consistent with prioritizing sensitivity in pre-screening.

Effect of β on Fβ score with fixed precision = 0.5: F₂ (recall-heavy) rises fastest with recall; F₁ is balanced; F₀.₅ is flattest.

All three curves meet at **recall = 0.5, Fβ = 0.5** when precision is also 0.5, because precision and recall are equal there.

#### 4.4.5 What validators store per day

For each eligible UID and day **E**, persist at least **Fβ**, and optionally **precision**, **recall**, and sample counts for audit. **Tier logic (§5) and ranking (§6)** use **Fβ** as the **primary metric** (daily value and its aggregate over lookback windows).

#### 4.4.6 Initial baseline model (reference)

The subnet problem is naturally **vision + language**: images (e.g. chest imaging) plus prompts or structured text, with the model producing a **text** output (e.g. classification rationale or a discrete label wrapped in generation). So the **reference starting point** is an **image–text–to–text** (vision-language) stack, not a single-modal classifier—though miners may later submit other architectures if they expose a compatible inference API.

**Documented baseline (example, not mandatory):** `[abersoz/Qwen2-VL-2B-Instruct-bnb-4bit-Radiology-mini](https://huggingface.co/abersoz/Qwen2-VL-2B-Instruct-bnb-4bit-Radiology-mini)` on Hugging Face. It is a **Qwen2-VL**–family model (image-text-to-text), fine-tuned from an **Instruct** 4-bit variant (~**2B parameters** per the model card), with naming and lineage that indicate **radiology-oriented** adaptation. Use that page for exact files, license (Apache-2.0 on the card), and revision history.

**Why this is only a baseline:** we **do not** claim this family or checkpoint is the **best** architecture for TB or broader pre-screening. It is a **practical anchor** for early subnet work: a public, reproducible artifact miners can fork, quantize, or replace. **Incentives are outcome-based (Fβ on time-shifted eval data)**—whatever miners commit and host on Chutes is what validators score. The baseline exists so teams share a **known-good starting recipe** while the open competition discovers stronger models.

### 4.5 Weight setting (incentives)

On each chain **tempo** (or subnet-specific weight update cadence), validators translate **accumulated tier emissions** (§5) into normalized weights and call `set_weights`. The current repo’s `validator.py` is a stub that assigns all weight to UID 0; production logic **replaces** this with tier-based distribution.

---

## 5. Incentive tiers (configurable, cumulative)

### 5.1 Principles

- **Tier table is fully configurable:** each tier specifies a **population rule** (e.g., top 1 globally, top 10% by count), a **lookback length in days**, and an **emission fraction** of total incentives for that tier.
- A single model (UID) can **qualify for multiple tiers** in the same weight-setting period. **Emissions add up** before normalization to that UID’s total share.
- **Anti early-deregistration:** models are evaluated **every day** once eligible; rolling windows use **daily stored scores**, not a one-shot event.
- **Anti early-registration gaming:** lower tiers reward sustained presence in upper percentiles over shorter lookbacks so new registrations still earn something if they perform well immediately, without replacing the dominant reward for long-horizon leadership.

### 5.2 Default tier sketch (all numbers configurable)

Illustrative defaults (replace entirely via config):


| Tier | Condition                                                             | Emission share |
| ---- | --------------------------------------------------------------------- | -------------- |
| A    | **Top 1** miner by §6 ranking over the **last 5** daily **Fβ** scores | **95%**        |
| B    | In **top 10%** by §6 ranking over the **last 4** days                 | **2%**         |
| C    | In **top 20%** over the **last 3** days                               | **1.5%**       |
| D    | In **top 30%** over the **last 2** days                               | **1%**         |
| E    | In **top 40%** over the **last 1** day                                | **0.5%**       |


**Percentiles:** “top p%” means the set of UIDs whose **aggregate Fβ** (or configured aggregate) over that lookback places them in the best p percent of active competitors (implementation defines handling of ties and minimum sample count per day).

**Window alignment:** “Last *k* days” means the *k* most recent **completed** daily evaluation records before the weight snapshot (or the *k* including today—implementation must fix one convention globally and document it in code/config).

**Normalization:** Configured tier percentages are targets; final weights must sum to the subnet’s allowed weight distribution after applying duplicate policy, inactive UIDs, and any burn/registrar rules.

### 5.3 Aggregating daily results for tiers

Each day produces one **daily Fβ** (and tie-break fields) per eligible UID. For a tier with lookback **k** days, validators compute an **aggregate score** over those **k** daily records (default: **arithmetic mean** of daily **Fβ** values). Tier membership (top 1, top p%, etc.) is determined from that aggregate and tie-breakers in §6. The aggregate function (e.g., mean vs min) should be **configurable**; **mean** is the default implied by “average of the last 5 days.”

### 5.4 Top model definition

The **headline “top model”** for the largest tier (default 95%) is the single UID that wins under §6 using the **aggregate over the last 5** daily **Fβ** results (default: mean **Fβ**), same rule as tier A. Shorter lookbacks and percentile tiers provide **additional** emission slices as configured.

---

## 6. Ranking and tie-breaking

**Daily:** each calendar day, UIDs are ordered by that day’s **Fβ** with the tie-breaks below.

**Tier / lookback:** for a window of **k** days, build each UID’s **aggregate metric** (default: mean of daily **Fβ**—§5.3), then order UIDs by that aggregate with the same tie-breaks:

1. **Primary:** higher **Fβ** (daily) or higher **aggregate Fβ** (lookback).
2. **Secondary:** **higher recall** (daily or same-window aggregate as primary—implementation must fix one). *Rationale:* if **Fβ** ties, prefer the model that misses fewer positives.
3. **Tertiary:** **higher precision** (same convention as secondary).
4. **Quaternary:** **smaller model size** wins (size metric fixed in commitment schema—e.g., parameter count or published bytes—and verified or trusted per subnet policy).
5. **Quinary:** **earlier submission** wins (earlier on-chain commit timestamp for the evaluated model version).

**Duplicates (§4.2):** only the **earliest-submitting** UID per duplicate group is in the ranking pool at all; **later duplicates are already ineligible**, so §6 tie-breaks apply only among **eligible** UIDs.

---

## 7. Configuration summary (validator-operated)


| Area                    | Configurable                                                                                                            |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Public dataset          | Base URL, query parameters, auth if any, task filters                                                                   |
| Temporal integrity      | Evaluation delay / cutoff after `T_sub`; timezone for “day” boundaries                                                  |
| Scoring                 | **β** for **Fβ**; definition of **positive vs negative** class; edge-case rules for empty denominators                  |
| Tier table              | For each tier: percentile or rank rule, lookback days, emission percentage, aggregate over daily **Fβ** (default: mean) |
| Duplicate policy        | How “same model” is detected; **first on-chain submitter eligible, all later duplicates ineligible**; deterministic tie-break if commit times collide |
| Chutes / request schema | Standard inference payload and **binary (or binarized) label mapping** for precision / recall / **Fβ**                  |


---

## 8. Repository boundary

- **In repo:** validator orchestration, chain reads, dataset client, Chutes client, daily scheduler, **Fβ** scoring (precision / recall from batch confusion counts), duplicate checks, weight derivation, persistence of daily results.
- **Out of repo:** miner training code, Hugging Face upload automation, Chutes packaging (miners perform these using their own tooling).

---

## 9. Relation to base `validator.py`

The existing `validator.py` connects to subtensor, syncs metagraph, and sets weights on a fixed tempo. The architecture above assumes this loop becomes the shell around: **commit parsing → duplicate filter (earliest submitter only per duplicate group) → daily evaluation job (inference → precision / recall → Fβ) → tier aggregation → `set_weights`**.