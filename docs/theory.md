# Theory and methods

Notation: pairwise judge `p(a,b)` = P(`a` in slot A beats `b`); pointwise score `s` on
`[lo, hi]`; human score `h`. Length effects are modelled in `log len` (words).

## Biases (null value in brackets)

- **Position**: `logit p(a,b) = f(a) − f(b) + β_pos`, with each pair judged in both orders.
  - `W = mean[(p(a,b) + p(b,a))/2]` [0.5]; `β̂ = mean[(logit p(a,b) + logit p(b,a))/2]` [0]
  - flip rate, and `d_z = mean(D)/sd(D)` with `D = p(a,b) + p(b,a) − 1`
- **Length**: filler padding to `pad_ratio = 2×` words, content unchanged.
  - padded win rate `mean[(p(c⁺,c) + 1 − p(c,c⁺))/2]` [0.5]
  - `β̂_len = mean[(logit p(c⁺,c) − logit p(c,c⁺)) / (2·log(len c⁺/len c))]`
  - pointwise `Δ = s(c⁺) − s(c)` [0]
- **Self-preference** (the judge's `family` appears in `source`):
  - own win rate [0.5], and the excess over human outcomes [0]
  - `L = β_self + γ·Δh` on order-averaged logits; pointwise `s = b0 + b1·h + b_self·own`
- **Scale compression**: `s = A + κ·(t − A)`. Fit `s = a + b·h`, then `κ̂ = b` [1] and
  the anchor is `a/(1−b)` (only if `|1−b| > 0.1`).

## Agreement and uncertainty

- Spearman, Pearson, MAE, and `κ_w = 1 − Σ w O / Σ w E`, with `w_ij = ((i−j)/(K−1))²`;
  pairwise accuracy and κ are given per order and order-averaged.
- Percentile cluster bootstrap by item (`n_boot = 1000`, `alpha = 0.05`). A result is
  flagged when its interval excludes the null.

## Corrections

- **Order averaging** (`PositionDebiased`): `p̃(a,b) = (p(a,b) + 1 − p(b,a))/2`.
- **Length control**: `s̃ = s − b1·(log len − mean log len)`, from OLS
  `s = b0 + b1·log len [+ b2·h]`. Pairwise, `σ(logit p − c1·log(len a/len b))`.
- **Calibration** (monotone maps to the human scale):
  - `isotonic`: pool-adjacent-violators
  - `platt`: `ŷ = lo + (hi−lo)·σ(a·z + b)`, fit by IRLS on `(h−lo)/(hi−lo)`
  - `linear`: OLS `ŷ = a + b·s`, clipped to the scale
  - item-grouped k-fold CV (`--folds 5`)

## Simulated judge (`FakeJudge`)

```
logit p = w·Δq + β_pos + β_len·log(len a/len b) + β_self·Δown + ε
s       = A + κ·(lo + q·(hi−lo) − A) + γ_len·log(len/L_ref) + γ_self·own + ε
```

`q` hashes the filler-free text, so padding leaves quality unchanged.

## References

- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
- Dubois et al. (2024). Length-Controlled AlpacaEval.
- Panickssery et al. (2024). LLM Evaluators Recognize and Favor Their Own Generations.
