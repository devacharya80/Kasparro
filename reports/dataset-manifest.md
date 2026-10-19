# Issued Dataset Manifest

The assessment specifies these version identifiers:
- Dataset: `long-brand-answers-50-v4`
- Catalogue: `speaker-catalogue-v4`
- Evaluator: `sentence-spans-and-labels-v4`

The SHA-256 values below were computed from the three separately supplied CSV uploads, read as bytes without normalization or re-saving.

| File | Rows | SHA-256 |
|---|---:|---|
| `answers.csv` | 50 | fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724 |
| `brand_facts.csv` | 132 | 2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2 |
| `development_gold.csv` | 376 | afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850 |

Use gold only after inference inside the development evaluator. Never send it to the model, tool executor, or inference prompt. The final-check split has no candidate-accessible labels and must not receive quality metrics.
