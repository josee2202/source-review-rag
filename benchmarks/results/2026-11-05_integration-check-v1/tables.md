<!-- tables:start -->
### Regression on the 30 known cases

| Reading | Correct | Macro-F1 | Wrongly supported |
|---|---|---|---|
| Raw label, before validation | 30/30 | 1.000 | 0/20 |
| Shown by the application | 29/30 | 0.982 | 0/20 |

Accepted 29/30; rejected by validation 1; technical failures 0; without a result 1/30.

### Control cases, fixed context

| Case | Tags | Expected | Raw label | Shown |
|---|---|---|---|---|
| IC-01 | days_60_90, number | contradicted | contradicted | contradicted |
| IC-02 | exceptions | supported | supported | supported |
| IC-03 | exceptions, scope | contradicted | contradicted | contradicted |
| IC-04 | multi_claim, correct_contradicted_unsupported | contradicted | contradicted | contradicted |
| IC-05 | multi_claim, correct_unsupported | insufficient_evidence | insufficient_evidence | insufficient_evidence |
| IC-06 | multi_claim, all_supported | supported | supported | no_valid_answer |
| IC-07 | irrelevant_omission | supported | supported | supported |
| IC-08 | simple_control | supported | supported | supported |
| IC-09 | simple_control, number | contradicted | contradicted | contradicted |
| IC-10 | simple_control, lexical_distractor | insufficient_evidence | insufficient_evidence | insufficient_evidence |

Shown correctly 9/10; rejected 1; technical failures 0.

### One document per mode, through the API

| Mode | Fragments | Shown correctly | Without analysis | With analysis | Judge only | First fragment | Median of the others |
|---|---|---|---|---|---|---|---|
| `text_vs_text` | 4 | 4/4 | 1.3 s | 121.3 s | 121.0 s | 35.0 s | 27.4 s |
| `report_vs_news` | 4 | 2/4 | 2.7 s | 144.5 s | 144.2 s | 37.8 s | 36.5 s |
| `news_vs_newsletter` | 6 | 6/6 | 1.8 s | 185.4 s | 185.0 s | 39.5 s | 30.5 s |

### Latency and memory of the fixed-context stages

Median per fragment 21.1 s, p95 26.7 s (n = 38, model already loaded). First request after loading: median 31.1 s (n = 2). Prompt median 928.5 tokens, output median 99.0 tokens. Maximum sampled RSS of the runner 6112.3 MiB; lowest free memory 67 %.
<!-- tables:end -->
