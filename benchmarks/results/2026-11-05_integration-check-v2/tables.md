<!-- tables:start -->
### Regression on the 30 known cases

| Reading | Correct | Macro-F1 | Wrongly supported |
|---|---|---|---|
| Raw label, before validation | 29/30 | 0.967 | 0/20 |
| Shown by the application | 28/30 | 0.965 | 0/20 |

Accepted 28/30; rejected by validation 2; technical failures 0; without a result 2/30.

### Control cases, fixed context

| Case | Tags | Expected | Raw label | Shown |
|---|---|---|---|---|
| IC-01 | days_60_90, number | contradicted | contradicted | contradicted |
| IC-02 | exceptions | supported | supported | supported |
| IC-03 | exceptions, scope | contradicted | contradicted | contradicted |
| IC-04 | multi_claim, correct_contradicted_unsupported | contradicted | contradicted | contradicted |
| IC-05 | multi_claim, correct_unsupported | insufficient_evidence | insufficient_evidence | insufficient_evidence |
| IC-06 | multi_claim, all_supported | supported | supported | supported |
| IC-07 | irrelevant_omission | supported | supported | supported |
| IC-08 | simple_control | supported | supported | supported |
| IC-09 | simple_control, number | contradicted | contradicted | contradicted |
| IC-10 | simple_control, lexical_distractor | insufficient_evidence | insufficient_evidence | insufficient_evidence |
| IC-11 | number, long_passage_with_headings | contradicted | contradicted | contradicted |
| IC-12 | long_passage_with_headings, lexical_distractor | insufficient_evidence | insufficient_evidence | insufficient_evidence |
| IC-13 | multi_claim, correct_and_contradicted, negation | contradicted | contradicted | contradicted |
| IC-14 | multi_claim, correct_contradicted_unsupported, cross_source | contradicted | contradicted | no_valid_answer |
| IC-15 | simple_control | supported | supported | supported |

Shown correctly 14/15; rejected 1; technical failures 0.

### One document per mode, through the API

| Mode | Fragments | Shown correctly | Without analysis | With analysis | Judge only | First fragment | Median of the others |
|---|---|---|---|---|---|---|---|
| `text_vs_text` | 4 | 4/4 | 1.5 s | 119.0 s | 118.7 s | 33.9 s | 28.0 s |
| `report_vs_news` | 4 | 3/4 | 2.5 s | 142.6 s | 142.4 s | 39.8 s | 35.9 s |
| `news_vs_newsletter` | 6 | 6/6 | 2.0 s | 176.9 s | 176.5 s | 37.5 s | 30.5 s |

### Latency and memory of the fixed-context stages

Median per fragment 23.8 s, p95 27.5 s (n = 43, model already loaded). First request after loading: median 30.9 s (n = 2). Prompt median 823 tokens, output median 104 tokens. Maximum sampled RSS of the runner 5982.1 MiB; lowest free memory 50 %.
<!-- tables:end -->
