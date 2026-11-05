# Integration check

Does the structured judge still work once it runs inside the real application? This page reports what happened when the production code
(`IA-modulable/juez.py`) was run with `qwen3.5:4b-q4_K_M` through the API, the interface and the Excel export. It comes after, and is separate
from, [the model comparison](model-comparison.md): one model, one pass, and no change to the frozen files or their results.

All cases are synthetic, and the assistant that ran the check (Claude Code) wrote and labelled them. **No person has reviewed the cases, the
labels or the explanations.** The results below do not show how the app behaves on real news, and the analysis stays off by default and
experimental.

## Result in short

- **Labels shown.** 28 of the 30 known cases and 14 of the 15 control cases were correct. The other three had no result: validation rejected them.
- **Full documents.** Through the API, 13 of 14 fragments in three documents were correct, and the other had no result.
- **Wrong supports.** None was shown in any stage: 0/20 and 0/10 on the negative cases with fixed context. With 20 cases, that still allows a true rate of up to about 16 %.
- **Technical failures.** None: no timeouts, cut-off answers or truncated prompts.
- **Time.** A document of 4 to 6 fragments took 2 to 3 minutes with the analysis and about 2 seconds without it, on an Intel laptop from 2019 with no GPU acceleration.
- **Mistakes that remain.** One validated explanation inverts the meaning of the exceptions (see [Review of the explanations](#review-of-the-explanations)). A literal quote does not protect against that.

## Two runs

The first run found a problem that the second one corrects. Both are kept, and neither was edited after running.

| | [`v1`](../benchmarks/results/2026-11-05_integration-check-v1/) | [`v2`](../benchmarks/results/2026-11-05_integration-check-v2/) |
|---|---|---|
| Commit | `92200b5` | `9892f6e` |
| Added to the evaluated prompt | Two longer rules: several claims, irrelevant omissions | One short rule: several claims |
| Quote | One piece | One or several pieces, each checked literally |
| Regression, correct as shown | 29/30 | 28/30 |
| Regression, raw label before validation | 30/30 | 29/30 |
| Controls, correct as shown | 9/10 | 14/15 |
| Documents, correct as shown | 12/14 | 13/14 |
| `report_vs_news` document, correct as shown | 2/4, including no result on 60 vs. 90 days | 3/4, with 60 vs. 90 days found |

**What went wrong in v1.** In the `report_vs_news` document, the answers for two fragments came back with no evidence. One of them was the 60 vs. 90 day
contrast. The frozen judge had answered both correctly, so the two rules added for this integration were the prime suspects. Validation did its job and
showed no label, but that was a regression.

**What changed.** On development cases and the failed fragments, the frozen prompt and a single short rule were compared. The frozen prompt, with no rule, gave a validated wrong answer on an invented mixed
paragraph and said the passages did not mention a price they did mention. The short rule withheld that answer and answered the failed fragments
correctly, so it was adopted. A second change came from a separate rejection: a supported paragraph needed evidence from two passages, and one quote could not
hold both, so a quote may now have several pieces. Five new control cases (IC-11 to IC-15) were fixed before v2 ran, because the first ten had already been seen. The logs are in
[`prompt_development/`](../benchmarks/results/2026-11-05_integration-check-v2/prompt_development/).

**v2 is not clearly better on the known cases.** It has one more rejection there, and the difference is too small to mean much. It was adopted for the full documents, on the
evidence above. The prompt was chosen after reading results, so only IC-11 to IC-15 were unseen by it.

## Results of v2

Each answer has two readings. The raw label is what the model wrote. "Shown" is what the app displays: only an answer that passes validation has a label, and a
rejection is never counted as correct, even if the raw label was right. `no_valid_answer` means no label was shown. All tables are generated from the saved results.

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

Requests used 8 threads on the 8 physical cores (`LLM_NUM_THREAD=8`). That was set for this check, and is not the default. Inference took 1,508 s in total, and the run completed with exit code 0.

Compared with the short passages of the fixed cases, the documents have longer passages, so their prompts are longer (about 900 to 1,120 tokens against a median of 823) and each fragment takes longer: the median across all 14 is 31.3 s, and the slowest
39.8 s, above the 30 s preference of the comparison. Without the analysis, the same requests took 1.5 to 2.5 s, including indexing a new reference.

### Answers with no result

| Case | Raw label | Expected | Reason for the rejection |
|---|---|---|---|
| EV-07 | supported | supported | The quote is correct, but it capitalises the first letter of a mid-sentence phrase. Also rejected in v1 |
| IC-14 | contradicted | contradicted | The same: a capitalised first letter |
| EV-27 | contradicted | insufficient_evidence | No passage and no quote. The raw label was wrong, and it was not shown |
| `report_vs_news` #4 | contradicted | insufficient_evidence | No passage and no quote. It treated the guide being fictitious as a contradiction. The raw label was wrong, and it was not shown |

In two of the four, validation held back a wrong label. In the other two, it discarded a right answer over a capital letter. The literal rule stays as it is: it is simple to explain, and
relaxing it after seeing these results would be tuning to them.

### The points this check had to cover

- **EV-24** (a name and a port appear, but no expansion): information insufficient in both runs, with an explanation saying that the expansion is not mentioned.
- **60 vs. 90 days.** Contradiction found with fixed context (IC-01) and through the API, naming both values, with the quote "más de 90 días de impago".
- **Exceptions for residents and people with reduced mobility.** Correct labels in IC-02, IC-03 and the document. The explanation in the document is wrong: see below.
- **A mixed paragraph** with correct, contradicted and unsupported claims. In the `text_vs_text` document, it says that 60 km is confirmed, 40 % conflicts with 25 %, and the tram is not mentioned. IC-04 gives the same three-way account, and IC-14 had the right raw label but was rejected.
- **Correct claim plus unsupported claim** (IC-05): information insufficient, without calling the unsupported part false.
- **Irrelevant omission** (IC-07): supported, and the omitted facts are not raised as a problem.

## Review of the explanations

The assistant, not a person, read the explanations of the 14 document fragments and the 15 controls of v2 against their passages. It found these concrete problems:

| Where | What it says | What the source says | Effect |
|---|---|---|---|
| `text_vs_text` #3, exceptions | The restriction "aplica a residentes y personas con movilidad reducida" | They are the exceptions to it | **Label and quote are right, the sentence inverts the meaning.** It was shown as a validated result |
| IC-04 | "seis profesionales de medicina general" | "seis consultas de medicina general" | Label right, one word wrong |
| IC-12 | The passages mention "cambios tarifarios" | "La tarifa no cambiará" | Label right, the summary is inaccurate |
| `text_vs_text` #4, mixed paragraph | Quotes the sentence about 60 km | The conflict is in the sentence with 25 % | The explanation is right, but the quote does not show the contradicted value |
| `report_vs_news` #4, not shown | The passages say the document is fictitious, so the claim is contradicted | Nothing about publishing models | Wrong reasoning, withheld by validation |

So a validated result shows that the quote exists, not that the sentence around it is right. That is why the interface always shows the original passages and asks people to read them.

Answers also vary. The explanation for `text_vs_text` #3 was correct when the same document was run from the browser ("salvo residentes o personas con movilidad reducida"), and `report_vs_news` #4 was validated correctly in a development run with the same prompt.
Temperature 0 and a fixed seed do not make runs identical here. The requests that came before differed, but the cause was not isolated.

## Interface and Excel

The real services were started over HTTP (`uvicorn` and Streamlit) and driven with headless Chrome, using `text_vs_text`, a temporary index and a separate browser profile. The scripts and the output are in
[`interface_check/`](../benchmarks/results/2026-11-05_integration-check-v2/interface_check/).

- The checkbox was off by default, and a click switched it on.
- The wait was 118.8 s for 4 fragments, with exactly 4 generations, all from `qwen3.5:4b-q4_K_M`, with `think=false` and 8 threads.
- For each fragment, the page and the downloaded workbook held the same label, explanation and quotes as the API response. The page kept the passages and showed the labels in natural language only.
- With a model that is not installed: "Análisis no disponible" on the 4 fragments, the `ollama pull` command in the message and the Excel file, the passages kept, no label, no download and no other model used.

For all three modes through the API, the check recomputes each result from the raw model answer and compares it with the response and the Excel file, and it counts one generation per fragment. All matched. With the analysis off, there were no requests to the model and no analysis
fields, and the passages were the same.

A fresh clone with a new virtual environment and the commands in the README passed `tests/smoke_e2e.py`, with and without `--con-analisis`, with exit code 0.

Only `text_vs_text` was driven in a browser. The other two modes share the same interface code, but were checked through the API.

## Limits of this check

- Small synthetic sets: 30, 15 and 14 fragments, one pass each. Figures are not rates to expect elsewhere.
- The 30 cases were used to choose the model, and the prompt was chosen after reading v1. Only IC-11 to IC-15 were unseen.
- Search quality is not evaluated: it was only checked that the needed passage was retrieved in these documents, and it was.
- Latency is from one machine, and memory is sampled every 10 s: the runner's maximum sampled RSS was 5,982 MiB. That figure includes mapped weights and is not the memory the system would recover.
- These results concern the analysis only. They do not validate a review as a whole, and a person still has to read the passages.

## Reproduce

```bash
ollama pull qwen3.5:4b-q4_K_M && ollama pull nomic-embed-text
.venv/bin/python benchmarks/run_integration_check.py --run-id my-check
.venv/bin/python benchmarks/build_integration_report.py --run-id my-check
```

See [`benchmarks/README.md`](../benchmarks/README.md). Answers can differ between runs, as noted above.
