# Chapter 09 — Scaling Laws and Emergent Behaviour

Empirical **scaling laws** predict how language-modelling loss moves with parameters \(N\), data \(D\), and compute \(C\). Kaplan (2020) popularised smooth **power-law** fits; Chinchilla (2022) revised **compute-optimal** data/model splits; emergence tracks sudden benchmark jumps as scale crosses thresholds—always controversial, but operationally useful for roadmaps.

```
   log(L − E)  ───── straight line ─────►  log N, log D, or log C
        │                              (until data mix, routing, or floor E bends fit)
        ▼
   planners translate elasticities into  N*, D*  at fixed $ / FLOPs budget
```

## Sub-chapters

**`01_Kaplan_Power_Laws/`**  
Single-variable laws \(\tilde{L}\propto N^{-\alpha_N}\), \(\propto D^{-\alpha_D}\), \(\propto C^{-\alpha_C}\) with \(\alpha_N\approx 0.076\), \(\alpha_D\approx 0.095\); irreducible floor \(E\); **joint** allocation discussion and contrast with Chinchilla iso-FLOP results.

**`02_Chinchilla/`**  
Fits compute–loss minima under matched FLOPs budgets; famously **\(D \approx 20 N\)** token-param ratio at optimality for many setups; reports fitted exponents for converting \(C\) to optimal \(N^\star,D^\star\) in \(\mathrm{TFLOP}\) ranges.

**`03_Emergent_Abilities/`** and **`03_Emergent_Capabilities/`** (duplicate naming)  
Surveys **sharp** vs **smooth** capability gains on benchmarks; **selection** concerns (benchmark contamination); separates **emergence as metric artefact** vs true phase-like transitions.

## Key formulas and concepts

**Kaplan-style (reduced loss)**

\[
L(N) - E \;\propto\; \Bigl(\frac{N_c}{N}\Bigr)^{\alpha_N},\quad
L(D) - E \;\propto\; \Bigl(\frac{D_c}{D}\Bigr)^{\alpha_D}.
\]

**Chinchilla-style allocation (illustrative)**

\[
N^\star \propto C^{a},\quad D^\star \propto C^{b},\quad a,b\approx 0.5\ \text{(reported fits)},\quad D/N \sim \mathcal{O}(10^1\text{–}10^2)\ \text{tokens/param}.
\]

**Compute proxy**

\[
C \approx 6 N D
\quad\text{(dense, no MoE, standard Adam counted separately).}
\]

**Elasticity mnemonic**

\[
\frac{\mathrm{d}\log \tilde{L}}{\mathrm{d}\log N} = -\alpha_N
\quad\Rightarrow\quad
\small \text{1\% more } N \Rightarrow \approx \alpha_N\%\ \text{reduction in }\tilde{L}\ (\text{locally}).
\]

## Prerequisites

- Log–log plots and power-law algebra.
- Chapter 06 (mapping real runs to \(N,D,C\)).

## Interpreting emergence responsibly

Large jumps on a benchmark can reflect **metric discontinuity**, **prompt formatting**, or **test contamination** rather than mystical novelty—pair emergence claims with **negative controls** (shuffle labels, trivial baselines).

## Study tips

- Replot public loss curves **with \(E\) subtracted** if known; slopes change near the floor.
- Cross-check iso-FLOP claims against **actual** training tokens **seen** vs advertised (dedup, epochs).

## Planning with laws (cautiously)

| Question | Law helps when… | Law fails when… |
|----------|----------------|-----------------|
| Next run budget order-of-magnitude | Same data mix + architecture family | New modality / tokenizer |
| Whether to add data vs width | Local log-log linear | Undertrained runs biased sweeps |
| Expected PPL drop from 2× FLOPs | Smooth continuum | Routing noise (MoE) changes \(C\) mapping |

## Ethics note

Scaling improves **average** helpfulness metrics but does not guarantee **fairness** or **truthfulness** without targeted mitigation—allocate eval beyond perplexity.

## Replication advice

- Publish **token counts** post-filtering **and** dedup policy; otherwise community cannot place your \(D\) on the same axis as Chinchilla-Kaplan plots.

## Intended outcomes

You should be able to: (1) translate a power-law statement into **doubling** predictions for loss; (2) compare single-resource vs joint **isoFLOP** reasoning; (3) explain why \(D/N\) ratios differ between Kaplan-era fits and Chinchilla recommendations; (4) critique an **emergence** claim using metric and contamination arguments.

## Exam-style checks

- Using \(\alpha_N=0.076\), compute the \(\tilde{L}\) ratio between \(N\) and \(2N\) **without** a calculator beyond \(2^{-0.076}\approx 0.95\).
- State one concrete reason Kaplan and Chinchilla **disagree on optimal \(D/N\)** that does **not** rely on vague “scale is always better” rhetoric.

**Mini extension:** Convert a reported Chinchilla-style **\(D/N\)** ratio into an implied **unique tokens per parameter** if you train exactly one epoch over a deduplicated pool—what breaks if you instead stream **infinite** repeats of the same batch?

Re-run the same \(\alpha_N\) doubling exercise using **perplexity** \(=\exp(L)\) instead of \(L\) to see how multiplicative loss changes translate to **geometric mean** token error.

When comparing models at different \(|\mathcal{V}|\), perplexities are only comparable with **identical tokenisation**—keep this caveat near any scaling-law chart.

---

_End of chapter overview._
