# Local model comparison

How three local models label a claim against given passages, and what each one costs on a CPU-only laptop. Tables and the
chart are generated from the saved results of one frozen run ([`benchmarks/results/2026-11-05_protocol-v1/`](../benchmarks/results/2026-11-05_protocol-v1/))
and one separate follow-up ([`benchmarks/results/2026-11-05_app-comments-followup/`](../benchmarks/results/2026-11-05_app-comments-followup/));
the text around them was written after reading those results.

> **Later update, not part of this comparison.** The text below is what was concluded at the time and is kept as written. Afterwards the
> owner chose `qwen3.5:4b-q4_K_M` as the configured model and replaced the free-text comment with the structured judge evaluated here,
> still off by default. The result of that work on the production code is in [the integration check](integration-check.md). The numbers
> on this page describe the benchmark's judge on 30 synthetic cases, not the application as a whole.

## Result in short

**If you turn comments on, use `qwen3.5:4b-q4_K_M`, and treat them as experimental.** Comments stay off by default. The default
model in the configuration has not been changed; [Decision](#decision) explains why.

- **`llama3.2:1b`** was the fastest and by far the weakest. It wrongly supported 14 of the 20 claims it should not have, and it never
  cited a passage.
- **`qwen3.5:4b-q4_K_M`** got 29/30 and then 30/30 right, with a median of 21.8 s per request on this set. It has one critical false
  support, in pass 1 only, so that problem is not resolved.
- **`qwen3.5:9b-q4_K_M`** had no false supports in either pass, got 27/30 and 28/30 right, and took almost twice as long (median 38.8 s),
  which is over the 30 s preference.

The rule fixed in advance names no robust winner between 4B and 9B, because their order flips between passes, and it then prefers the
lower cost. These are 30 tidy synthetic cases written by the assistant that ran the comparison, with no human review. High scores on
them do not show that a model is reliable on real news.

<!-- BEGIN GENERATED: overview -->
| Model | Valid answers | Correct / attempted | 95 % CI (Wilson) | Macro-F1 | False supports / negatives | Critical false supports / critical cases | Warm median | Warm p95 |
|---|---|---|---|---|---|---|---|---|
| **1B** `llama3.2:1b` | 30/30 | 8/30 | 14–44 % | 0.24 | 14/20 | 13/18 | 8.1 s (n=29) | 12.2 s |
| **4B** `qwen3.5:4b-q4_K_M` | 30/30 | 29/30 | 83–99 % | 0.97 | 1/20 | 1/18 | 21.8 s (n=29) | 24.9 s |
| **9B** `qwen3.5:9b-q4_K_M` | 30/30 | 27/30 | 74–97 % | 0.90 | 0/20 | 0/18 | 38.8 s (n=29) | 47.5 s |
<!-- END GENERATED: overview -->

Valid answers count over the 30 planned cases. Failures count as not correct. "Negatives" are the 20 cases whose expected
label is not `supported`. Warm times exclude the first request after each load.

![Macro-F1 against median warm time for the three models. The same values are in the table above.](figures/model-comparison.svg)

## Why this comparison

The optional comment in Source Review RAG used `llama3.2:1b` and had never been measured. In earlier manual checks it had
missed a 60-versus-90-day discrepancy and invented figures and entities, but that was an anecdote, so this run starts again
from explicit cases. The question is practical: is a larger local model worth its cost on this machine, or should
comments stay experimental? The two candidates are `qwen3.5:4b-q4_K_M` and `qwen3.5:9b-q4_K_M`. The larger model was not
expected to win by default, and "none of them" was an accepted outcome from the start.

This is not a general fact-checking evaluation. A label only says how a claim relates to the passages supplied.

## Method

The protocol, the datasets and the selection rule were hashed in [`benchmarks/protocol.lock.json`](../benchmarks/protocol.lock.json)
and pushed in commit `1af7098` before any model answered an evaluation case. Six separate development cases were used to fix format
and integration problems; what changed is recorded in [`benchmarks/protocol.yaml`](../benchmarks/protocol.yaml). The only use of
evaluation prompts during development was counting the tokens of the three longest, with the output discarded. The prompt was not
touched after the freeze.

**Data.** 30 evaluation cases, ten per expected label (`supported`, `contradicted`, `insufficient_evidence`), across the three
modes: 9 `text_vs_text`, 12 `report_vs_news`, 9 `news_vs_newsletter`. They cover figures, units, percentages, dates, 60/90-day
deadlines, negations, attribution to the wrong entity, the scope of a claim, lexically similar but irrelevant evidence, legitimate
paraphrases and one injected instruction. Each of the ten fictional scenarios supplies three claims over the same passages, so
**cases are not fully independent**. Insufficient evidence means insufficient within the supplied passages only.

**The data and labels were written and reviewed by the assistant that ran this comparison (Claude Code). They have had no human
review.** This is not an independent benchmark, not a human validation, and not a representative sample of real news. None of
the evaluated models judged any answer, and no external LLM was used as ground truth: answers are scored mechanically against the
labels in the dataset.

**A. Quality with fixed evidence.** The three models receive byte-identical prompts: same claim, same passages in the same
order, no retrieval involved, which isolates the judge from search failures. Each must return JSON with passage IDs, a brief
literal quote, a one- or two-sentence explanation and a label. Expected answers never reach the prompt, which a test verifies.
Two passes run over all the cases in a reproducible order, with the model order reversed in the second pass. The second pass
measures repeatability and adds no independent cases. **Tables use pass 1**; pass 2 only tests whether the outcome is stable.

**B. The full pipeline.** One example per mode goes through the real API code with the existing chunking, embeddings and top-3
search unchanged. Each fragment is judged on what was really retrieved, so a search failure can be told apart from a judge failure.
The app's own comment path, with its prompt and adapter as they were at the frozen commit, runs on the same documents. This is a
limited integration check and not a retrieval evaluation.

**Request.** All three use the same explicit parameters: `temperature` 0, `seed` 42, `top_k` 40, `top_p` 0.9, no repeat, presence or
frequency penalty, `num_ctx` 4096, `num_predict` 256, `num_thread` 8, `think: false`, and the same JSON schema through `format`. These
override what Qwen3.5 inherits from its model file (`presence_penalty` 1.5, `temperature` 1, `top_k` 20, `top_p` 0.95), and thread
count was not tuned. With temperature 0 the second pass mostly reveals nondeterminism rather than sampling variation. Each
model's official template was respected. The timeout was 120 s per request, with no retries.

**What the metrics mean.**

- *False support:* predicting `supported` when the label is `contradicted` or `insufficient_evidence`, over the negatives attempted.
  It is the error this tool can least afford.
- *Critical false support:* the same error on a case marked critical in advance, which means every contradicted case plus the
  insufficient-evidence cases built around a similar distractor or an injected instruction.
- *Traceability* is mechanical: cited IDs exist, the quote appears literally in a cited passage, the expected passage is cited.
  **A valid citation does not prove that the explanation is right. The semantic fidelity of the explanations is pending human review**,
  and no semantic hallucination rate is reported.
- *Figures or capitalised words absent from the inputs* are simple text checks, a proxy for invented data. The capitalised-word one
  is crude: it can flag harmless words and it misses invented lower-case terms.
- *Wall time* is what the person waits for. It is not replaced by tokens per second. Time to first token is measured through
  streaming.
- *Memory:* resident size of the runner, sampled every 5 s, so "max sampled" is not an exact peak. RSS includes file-mapped weights
  and may include shared pages, so it is neither the weights file size nor what the system would recover. Runners are never
  added up. Energy was not measured.

**Selection rule, fixed in advance.** Among complete candidates, fewer false supports win, then higher macro-F1. If the gap is a
single case and critical errors do not change, or if the order flips between passes, there is no robust winner and lower cost is
preferred. A median warm time of up to 30 s per fragment is a product preference, not a prediction or a guarantee. Any unresolved
critical false support keeps comments off by default.

## Environment

<!-- BEGIN GENERATED: environment -->
- **Machine:** Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz, 8 cores / 16 threads, 16 GiB RAM, macOS 14.5 (23F79). The system also reports Intel UHD Graphics 630 and AMD Radeon Pro 5500M; Ollama did not use them in this run.
- **Ollama:** 0.40.14. Backend according to its log: `library=cpu`; `/api/ps` reported `size_vram` = 0 on every load, which is what `ollama ps` shows as 100 % CPU.
- **Run:** `2026-11-05_protocol-v1`, protocol v1, commit `1af70980aa31`, frozen: yes. From 2026-11-05T10:25:31+00:00 to 2026-11-05T13:02:08+00:00, resumed 0 time(s).
- **Model execution time:** 2.602 h of the 6 h budget. Server log lines mentioning truncation during the run: 0.

| Alias | Tag | Digest | Family | Parameters | Quantization | Model context limit | Capabilities | Inherited parameters (overridden) | Prompt format |
|---|---|---|---|---|---|---|---|---|---|
| 1B | `llama3.2:1b` | `baf6a787fdff` | llama | 1.2B | Q8_0 | 131072 | completion, tools | none | template `966de95ca8a6` |
| 4B | `qwen3.5:4b-q4_K_M` | `d8b0f5e9760c` | qwen35 | 4.2B | Q4_K_M | 262144 | completion, vision, tools, thinking | presence_penalty 1.5; temperature 1; top_k 20; top_p 0.95 | renderer qwen3.5 |
| 9B | `qwen3.5:9b-q4_K_M` | `56671c2ab938` | qwen35 | 9.0B | Q4_K_M | 262144 | completion, vision, tools, thinking | presence_penalty 1.5; temperature 1; top_k 20; top_p 0.95 | renderer qwen3.5 |
<!-- END GENERATED: environment -->

Other applications stayed open, as on a normal working day, and were not closed for the run. Cold load means that the model was not in
Ollama's memory; the operating system's file cache may still have held its weights, so it can be faster than a first load after boot.

### Pilot

Three development cases per model before the evaluation, plus a check that aborting a long generation really stops the runner.

<!-- BEGIN GENERATED: pilot -->
| Model | Pilot statuses | Pilot wall (s) | Cancel check | Longest eval prompt, tokens | Decision |
|---|---|---|---|---|---|
| 1B | ok, ok, ok | 11.4 / 8.9 / 7.0 | stopped in 4.5 s (64.4 CPU s used before cancelling) | 875 ≤ 3840 | viable |
| 4B | ok, ok, ok | 21.3 / 22.6 / 14.9 | stopped in 4.5 s (64.5 CPU s used before cancelling) | 803 ≤ 3840 | viable |
| 9B | ok, ok, ok | 39.9 / 39.3 / 45.4 | stopped in 4.5 s (63.7 CPU s used before cancelling) | 803 ≤ 3840 | viable |
<!-- END GENERATED: pilot -->

## A. Quality with fixed evidence

### Failures by type

<!-- BEGIN GENERATED: failures_by_type -->
| Model | Attempted | Valid | Format errors | Truncated | Timeouts | Request errors | Not executed | Completion |
|---|---|---|---|---|---|---|---|---|
| 1B | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 100 % |
| 4B | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 100 % |
| 9B | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 100 % |
<!-- END GENERATED: failures_by_type -->

### Confusion matrices

<!-- BEGIN GENERATED: confusion -->
**1B** (pass 1; rows are expected, columns predicted)

| expected \ predicted | supported | contradicted | insufficient_evidence | no_valid_answer |
|---|---|---|---|---|
| **supported** | 5 | 0 | 5 | 0 |
| **contradicted** | 6 | 1 | 3 | 0 |
| **insufficient_evidence** | 8 | 0 | 2 | 0 |

**4B** (pass 1; rows are expected, columns predicted)

| expected \ predicted | supported | contradicted | insufficient_evidence | no_valid_answer |
|---|---|---|---|---|
| **supported** | 10 | 0 | 0 | 0 |
| **contradicted** | 0 | 10 | 0 | 0 |
| **insufficient_evidence** | 1 | 0 | 9 | 0 |

**9B** (pass 1; rows are expected, columns predicted)

| expected \ predicted | supported | contradicted | insufficient_evidence | no_valid_answer |
|---|---|---|---|---|
| **supported** | 9 | 0 | 1 | 0 |
| **contradicted** | 0 | 10 | 0 | 0 |
| **insufficient_evidence** | 0 | 2 | 8 | 0 |
<!-- END GENERATED: confusion -->

### Precision and recall by class

Failed requests count against recall. F1 is 0 when precision or recall is undefined.

<!-- BEGIN GENERATED: per_class -->
| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| 1B | supported | 0.26 (5/19) | 0.50 (5/10) | 0.34 |
| 1B | contradicted | 1.00 (1/1) | 0.10 (1/10) | 0.18 |
| 1B | insufficient_evidence | 0.20 (2/10) | 0.20 (2/10) | 0.20 |
| 4B | supported | 0.91 (10/11) | 1.00 (10/10) | 0.95 |
| 4B | contradicted | 1.00 (10/10) | 1.00 (10/10) | 1.00 |
| 4B | insufficient_evidence | 1.00 (9/9) | 0.90 (9/10) | 0.95 |
| 9B | supported | 1.00 (9/9) | 0.90 (9/10) | 0.95 |
| 9B | contradicted | 0.83 (10/12) | 1.00 (10/10) | 0.91 |
| 9B | insufficient_evidence | 0.89 (8/9) | 0.80 (8/10) | 0.84 |
<!-- END GENERATED: per_class -->

### Correct answers by case property

A case can have several properties, and most counts are very small.

<!-- BEGIN GENERATED: by_tag -->
| Case property | 1B | 4B | 9B |
|---|---|---|---|
| `absent` | 1/3 | 3/3 | 3/3 |
| `cross_source` | 0/3 | 3/3 | 3/3 |
| `date` | 0/2 | 2/2 | 2/2 |
| `deadline_60_90` | 0/2 | 2/2 | 2/2 |
| `evidence_first` | 0/1 | 1/1 | 1/1 |
| `figure` | 1/8 | 8/8 | 7/8 |
| `lexical_distractor` | 2/11 | 10/11 | 9/11 |
| `negation` | 3/6 | 6/6 | 6/6 |
| `paraphrase` | 4/9 | 9/9 | 8/9 |
| `percentage` | 2/4 | 4/4 | 4/4 |
| `prompt_injection` | 0/1 | 1/1 | 1/1 |
| `scope` | 0/1 | 1/1 | 1/1 |
| `simple_sum` | 0/1 | 1/1 | 0/1 |
| `unit` | 0/1 | 1/1 | 1/1 |
| `wrong_entity` | 0/4 | 4/4 | 4/4 |
<!-- END GENERATED: by_tag -->

### Where the models disagree

Cases that only one model of a pair got right, in pass 1. With this few cases these counts describe this run; they are not a
significance test.

<!-- BEGIN GENERATED: discordant -->
| Pair | Shared cases | Only the first right | Only the second right |
|---|---|---|---|
| 1B vs 4B | 30 | 0 | 21 |
| 1B vs 9B | 30 | 0 | 19 |
| 4B vs 9B | 30 | 2 | 0 |
<!-- END GENERATED: discordant -->

### Traceability

<!-- BEGIN GENERATED: traceability -->
| Model | Answers citing a nonexistent passage ID | Non-empty quote found literally in a cited passage | Expected passage cited | Both values named in the explanation | Correct label + expected passage + literal quote | Explanations with a figure absent from the inputs | Explanations with a capitalised word absent from the inputs |
|---|---|---|---|---|---|---|---|
| 1B | 0/30 | 0/18 | 0/20 | 0/6 | 2/30 | 0/30 | 0/30 |
| 4B | 0/30 | 20/21 | 20/20 | 6/6 | 28/30 | 1/30 | 0/30 |
| 9B | 0/30 | 20/22 | 20/20 | 6/6 | 25/30 | 0/30 | 0/30 |
<!-- END GENERATED: traceability -->

`1B` returned an empty `passage_ids` list in all 60 answers across both passes. The schema allows an empty list, because
`insufficient_evidence` may need one, so this is how it behaved under this format and not a crash. In pass 1, 10 of its 18 non-empty
quotes simply repeated the claim.

The non-literal quotes of the other two models are small, and the strict check counts them anyway: in `EV-07` both capitalise a quote
taken from mid-sentence, and in `EV-22` the 9B joins two sentences with an ellipsis. The one figure flagged for the 4B is the divisor in
"1,50 euros / 2", which is arithmetic and not invented data.

### Examples of failures

Literal output from pass 1: critical false supports if the model has any, otherwise any other error.

<!-- BEGIN GENERATED: failure_examples -->
**1B** (13 in this category, first 3 shown)

- `EV-02` · claim: «El ayuntamiento dispone de hasta 90 días para resolver las solicitudes de subvención del contenedor.» · expected `contradicted`, got `supported`. Quote: «El ayuntamiento dispone de hasta 90 días para resolver las solicitudes de subvención del contenedor.». Explanation: «El fragmento dice que el ayuntamiento dispone de hasta 90 días para resolver las solicitudes de subvención del contenedor.»
- `EV-05` · claim: «La Biblioteca Central ampliará su horario a partir del 3 de octubre de 2025.» · expected `contradicted`, got `supported`. Quote: «». Explanation: «El fragmento dice que la Biblioteca Central ampliará su horario a partir del 3 de septiembre de 2025.»
- `EV-06` · claim: «La Biblioteca Central prestará portátiles a sus usuarios.» · expected `insufficient_evidence`, got `supported`. Quote: «». Explanation: «El fragmento dice que la Biblioteca Central prestará portátiles a sus usuarios.»

**4B** (1 in this category, first 1 shown)

- `EV-24` · claim: «Marta Ilundain anunció la ampliación del Puerto de Bahía Clara.» · expected `insufficient_evidence`, got `supported`. Quote: «Ilundain agradeció el trabajo de los estibadores y de las empresas consignatarias durante la presentación de los datos.». Explanation: «El pasaje P2 confirma que Marta Ilundain presentó datos sobre el Puerto de Bahía Clara, lo cual respalda implícitamente que anunció ampliaciones o mejoras relacionadas con el puerto.»

**9B** (3 in this category, first 3 shown)

- `EV-04` · claim: «Cuando acabe la obra, la Biblioteca Central tendrá 500 puestos de lectura.» · expected `supported`, got `insufficient_evidence`. Quote: «La obra añadirá 180 puestos de lectura a los 320 actuales». Explanation: «El fragmento afirma que habrá 500 puestos tras la obra, pero el pasaje P3 indica que se añadirán 180 a los 320 actuales, lo que suma 500; sin embargo, el pasaje no especifica si los 320 puestos actuales son los únicos existentes o si la cifra final es exactamente 500, por lo que la evidencia es insuficiente para confirmar la afirmación absoluta del fragmento sin asumir que no hay otros puestos.»
- `EV-24` · claim: «Marta Ilundain anunció la ampliación del Puerto de Bahía Clara.» · expected `insufficient_evidence`, got `contradicted`. Quote: «El Puerto de Bahía Clara movió 1,8 millones de contenedores en 2024, un récord histórico, según informó su presidenta, Marta Ilundain.». Explanation: «El pasaje P2 indica que Marta Ilundain informó sobre un récord histórico de tráfico de contenedores, mientras que el fragmento afirma que ella anunció una ampliación del puerto; no hay evidencia en los pasajes de que se haya anunciado una ampliación, por lo que la información es incompatible.»
- `EV-27` · claim: «El precio del aceite de oliva bajará la próxima primavera.» · expected `insufficient_evidence`, got `contradicted`. Quote: «Los operadores prevén otra subida a lo largo del invierno por la escasez de existencias.». Explanation: «El pasaje P2 indica que se espera un aumento de precios en el invierno debido a la falta de stock, lo cual contradice la afirmación del fragmento sobre una bajada de precios en la primavera.»
<!-- END GENERATED: failure_examples -->

### Repeatability

<!-- BEGIN GENERATED: repeatability -->
| Model | Same label | Identical text | Correct (pass 1 / 2) | False supports (1 / 2) | Macro-F1 (1 / 2) | Warm median s (1 / 2) | Label changes |
|---|---|---|---|---|---|---|---|
| 1B | 28/30 | 27/30 | 8/30 / 9/30 | 14/20 / 13/20 | 0.24 / 0.27 | 8.1 / 8.3 | EV-01 (insufficient_evidence → contradicted), EV-15 (supported → insufficient_evidence) |
| 4B | 29/30 | 11/30 | 29/30 / 30/30 | 1/20 / 0/20 | 0.97 / 1.00 | 21.8 / 20.6 | EV-24 (supported → insufficient_evidence) |
| 9B | 28/30 | 14/30 | 27/30 / 28/30 | 0/20 / 0/20 | 0.90 / 0.95 | 38.8 / 39.1 | EV-04 (insufficient_evidence → no_valid_answer), EV-24 (contradicted → insufficient_evidence) |
<!-- END GENERATED: repeatability -->

Labels held up better than text. Although temperature is 0 and the seed is fixed, the wording often changed between passes. Cases ran
in a different order, so a different part of each prompt came from Ollama's cache, which may explain it; this was not tested.

`EV-24` is the case to watch. It claims that a named person announced a port expansion, while the passages only have her presenting
traffic figures. The same prompt produced `supported` and then `insufficient_evidence` from the 4B, and `contradicted` and then
`insufficient_evidence` from the 9B. The 9B's failure in pass 2 on `EV-04` is a truncation: it reached the limit of 256 tokens.

## Time and memory

<!-- BEGIN GENERATED: performance -->
| Model | Pass | Completed | Timeouts | Total request time | Cold load (s) | First request after load (s) | Warm wall median / p95 (s) | Warm time to first token (s) | Warm prompt eval (s) | Warm generation (s) | Prompt tokens / cached (median) | Output tokens (median) | Generation tok/s (median) | CPU speed limit, median / min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1B | 1 | 30 | 0 | 4.1 min | 1.8 | 11.0 | 8.1 / 12.2 | 3.6 | 3.6 | 4.5 | 848 / 334 | 83 | 18.2 | 100 / 100 % |
| 1B | 2 | 30 | 0 | 4.3 min | 1.8 | 10.7 | 8.3 / 13.3 | 3.5 | 3.5 | 4.7 | 848 / 334 | 84 | 18.1 | 100 / 100 % |
| 4B | 1 | 30 | 0 | 10.6 min | 6.1 | 28.7 | 21.8 / 24.9 | 6.9 | 6.7 | 14.1 | 789 / 291 | 104 | 7.2 | 100 / 62 % |
| 4B | 2 | 30 | 0 | 10.7 min | 5.9 | 29.7 | 20.6 / 25.5 | 6.7 | 6.5 | 14.0 | 789 / 291 | 103 | 7.3 | 100 / 62 % |
| 9B | 1 | 30 | 0 | 20.0 min | 15.9 | 52.5 | 38.8 / 47.5 | 14.1 | 13.7 | 25.1 | 789 / 291 | 104 | 4.1 | 100 / 62 % |
| 9B | 2 | 30 | 0 | 20.0 min | 15.1 | 56.5 | 39.1 / 49.7 | 13.5 | 13.3 | 24.5 | 789 / 291 | 104 | 4.1 | 100 / 64 % |
<!-- END GENERATED: performance -->

Prompts share a prefix, and Ollama reuses it from its cache between requests, which is why warm prompt evaluation is shorter than the
first request after a load. Prompts in section B are longer and most of them reuse no prefix, so section B is the better guide to real
waiting time.

<!-- BEGIN GENERATED: memory -->
| Model | Files on disk (all layers) | Weights file | Size in `/api/ps` | Runner RSS median / max sampled | System memory free, median / min | Swap used, first → last sample | Samples at warn / critical pressure | Lowest CPU speed limit | `size_vram` |
|---|---|---|---|---|---|---|---|---|---|
| 1B | 1.32 GB | 1.32 GB | 1.52 GB | 1761 / 1975 MiB | 74 / 73 % | 4900 → 3940 MiB | 0 / 0 of 107 | 100 % | 0 |
| 4B | 3.32 GB | 2.65 GB | 3.15 GB | 4936 / 6506 MiB | 74 / 61 % | 4580 → 4196 MiB | 0 / 0 of 264 | 62 % | 0 |
| 9B | 6.55 GB | 5.63 GB | 6.19 GB | 6635 / 7601 MiB | 71 / 39 % | 4452 → 5187 MiB | 2 / 0 of 501 | 62 % | 0 |
<!-- END GENERATED: memory -->

## B. The full pipeline

Whole-document times, including embeddings and search. A new index includes loading the embedding model. "App comment path" is the
application at the frozen commit: its own prompt, free text, no `think` option and a cold generative model on the first fragment.

<!-- BEGIN GENERATED: e2e_documents -->
| Model | Mode | Fragments | New index: total (index + search), s | Reused index: total (index + search), s | Structured judge, whole document (first / warm median), s | App comment path, whole request (first / warm median), s | Non-empty app comments |
|---|---|---|---|---|---|---|---|
| 1B | text vs text | 4 | 1.2 (0.9 + 0.3) | 0.2 (0.00 + 0.2) | 44.8 (12.9 / 10.1) | 64.1 (20.5 / 14.3) | 4/4 |
| 1B | report vs news | 4 | 2.0 (1.6 + 0.2) | 0.2 (0.00 + 0.2) | 46.0 (11.5 / 12.3) | 92.3 (23.5 / 25.2) | 4/4 |
| 1B | news vs newsletter | 6 | 1.2 (0.9 + 0.3) | 0.3 (0.00 + 0.3) | 63.6 (12.3 / 11.2) | 147.5 (35.0 / 23.1) | 6/6 |
| 4B | text vs text | 4 | 1.3 (1.0 + 0.3) | 0.3 (0.00 + 0.3) | 98.6 (24.4 / 25.6) | 331.8 (88.1 / 83.6) | 0/4 |
| 4B | report vs news | 4 | 2.2 (2.0 + 0.2) | 0.2 (0.00 + 0.2) | 167.3 (46.8 / 42.7) | 336.7 (90.0 / 82.3) | 0/4 |
| 4B | news vs newsletter | 6 | 1.5 (1.2 + 0.3) | 0.3 (0.00 + 0.3) | 145.3 (28.5 / 25.1) | 471.7 (85.4 / 78.8) | 0/6 |
| 9B | text vs text | 4 | 1.3 (1.0 + 0.3) | 0.3 (0.00 + 0.3) | 171.9 (42.4 / 41.8) | 541.5 (149.5 / 130.3) | 0/4 |
| 9B | report vs news | 4 | 2.0 (1.8 + 0.2) | 0.2 (0.00 + 0.2) | 210.4 (53.5 / 52.0) | 575.3 (154.5 / 140.8) | 0/4 |
| 9B | news vs newsletter | 6 | 1.5 (1.2 + 0.3) | 0.3 (0.00 + 0.3) | 271.0 (50.1 / 45.2) | 790.6 (150.4 / 128.2) | 0/6 |
<!-- END GENERATED: e2e_documents -->

### Search failure or judge failure

<!-- BEGIN GENERATED: e2e_outcomes -->
| Model | Mode | Needed passage in top 3 | Correct label | Judge errors | Retrieval misses | Failed requests |
|---|---|---|---|---|---|---|
| 1B | text vs text | 4/4 | 2/4 | 2 | 0 | 0 |
| 1B | report vs news | 3/3 | 1/4 | 3 | 0 | 0 |
| 1B | news vs newsletter | 5/5 | 5/6 | 1 | 0 | 0 |
| 4B | text vs text | 4/4 | 4/4 | 0 | 0 | 0 |
| 4B | report vs news | 3/3 | 4/4 | 0 | 0 | 0 |
| 4B | news vs newsletter | 5/5 | 6/6 | 0 | 0 | 0 |
| 9B | text vs text | 4/4 | 4/4 | 0 | 0 | 0 |
| 9B | report vs news | 3/3 | 4/4 | 0 | 0 | 0 |
| 9B | news vs newsletter | 5/5 | 6/6 | 0 | 0 | 0 |
<!-- END GENERATED: e2e_outcomes -->

<!-- BEGIN GENERATED: e2e_fragments -->
| Model | Mode | # | Expected | Predicted | Needed passage retrieved | Values named | Outcome | Wall s |
|---|---|---|---|---|---|---|---|---|
| 1B | text vs text | 1 | `supported` | `supported` | yes | n/a | correct | 12.9 |
| 1B | text vs text | 2 | `contradicted` | `supported` | yes | no | judge_error | 13.2 |
| 1B | text vs text | 3 | `supported` | `supported` | yes | n/a | correct | 10.1 |
| 1B | text vs text | 4 | `contradicted` | `supported` | yes | no | judge_error | 8.6 |
| 1B | report vs news | 1 | `contradicted` | `insufficient_evidence` | yes | no | judge_error | 11.5 |
| 1B | report vs news | 2 | `supported` | `contradicted` | yes | n/a | judge_error | 14.1 |
| 1B | report vs news | 3 | `supported` | `insufficient_evidence` | yes | n/a | judge_error | 12.3 |
| 1B | report vs news | 4 | `insufficient_evidence` | `insufficient_evidence` | n/a | n/a | correct | 8.1 |
| 1B | news vs newsletter | 1 | `supported` | `supported` | yes | n/a | correct | 12.3 |
| 1B | news vs newsletter | 2 | `supported` | `supported` | yes | n/a | correct | 10.0 |
| 1B | news vs newsletter | 3 | `contradicted` | `contradicted` | yes | no | correct | 11.2 |
| 1B | news vs newsletter | 4 | `contradicted` | `supported` | yes | n/a | judge_error | 11.5 |
| 1B | news vs newsletter | 5 | `supported` | `supported` | yes | n/a | correct | 11.5 |
| 1B | news vs newsletter | 6 | `insufficient_evidence` | `insufficient_evidence` | n/a | n/a | correct | 7.2 |
| 4B | text vs text | 1 | `supported` | `supported` | yes | n/a | correct | 24.4 |
| 4B | text vs text | 2 | `contradicted` | `contradicted` | yes | yes | correct | 25.6 |
| 4B | text vs text | 3 | `supported` | `supported` | yes | n/a | correct | 25.7 |
| 4B | text vs text | 4 | `contradicted` | `contradicted` | yes | yes | correct | 22.9 |
| 4B | report vs news | 1 | `contradicted` | `contradicted` | yes | yes | correct | 46.8 |
| 4B | report vs news | 2 | `supported` | `supported` | yes | n/a | correct | 51.3 |
| 4B | report vs news | 3 | `supported` | `supported` | yes | n/a | correct | 42.7 |
| 4B | report vs news | 4 | `insufficient_evidence` | `insufficient_evidence` | n/a | n/a | correct | 26.5 |
| 4B | news vs newsletter | 1 | `supported` | `supported` | yes | n/a | correct | 28.5 |
| 4B | news vs newsletter | 2 | `supported` | `supported` | yes | n/a | correct | 25.1 |
| 4B | news vs newsletter | 3 | `contradicted` | `contradicted` | yes | yes | correct | 22.0 |
| 4B | news vs newsletter | 4 | `contradicted` | `contradicted` | yes | n/a | correct | 25.2 |
| 4B | news vs newsletter | 5 | `supported` | `supported` | yes | n/a | correct | 26.5 |
| 4B | news vs newsletter | 6 | `insufficient_evidence` | `insufficient_evidence` | n/a | n/a | correct | 18.0 |
| 9B | text vs text | 1 | `supported` | `supported` | yes | n/a | correct | 42.4 |
| 9B | text vs text | 2 | `contradicted` | `contradicted` | yes | yes | correct | 47.4 |
| 9B | text vs text | 3 | `supported` | `supported` | yes | n/a | correct | 40.3 |
| 9B | text vs text | 4 | `contradicted` | `contradicted` | yes | yes | correct | 41.8 |
| 9B | report vs news | 1 | `contradicted` | `contradicted` | yes | yes | correct | 53.5 |
| 9B | report vs news | 2 | `supported` | `supported` | yes | n/a | correct | 54.6 |
| 9B | report vs news | 3 | `supported` | `supported` | yes | n/a | correct | 52.0 |
| 9B | report vs news | 4 | `insufficient_evidence` | `insufficient_evidence` | n/a | n/a | correct | 50.4 |
| 9B | news vs newsletter | 1 | `supported` | `supported` | yes | n/a | correct | 50.1 |
| 9B | news vs newsletter | 2 | `supported` | `supported` | yes | n/a | correct | 48.4 |
| 9B | news vs newsletter | 3 | `contradicted` | `contradicted` | yes | yes | correct | 36.6 |
| 9B | news vs newsletter | 4 | `contradicted` | `contradicted` | yes | n/a | correct | 45.2 |
| 9B | news vs newsletter | 5 | `supported` | `supported` | yes | n/a | correct | 57.9 |
| 9B | news vs newsletter | 6 | `insufficient_evidence` | `insufficient_evidence` | n/a | n/a | correct | 32.9 |
<!-- END GENERATED: e2e_fragments -->

**Search caused no failure here.** The passage needed was in the top 3 for all 12 fragments that needed one. These are three small
documents, so this says nothing about retrieval in general.

**The labels follow section A.** The 4B and the 9B labelled all 14 fragments as expected, and the 1B 8 of 14. In the paragraph with
three claims (`text vs text`, #4), the 1B answered `supported` by repeating the paragraph. Both Qwen3.5 models answered `contradicted`
and pointed to the 25 % against 40 % figure and the missing tram line.

**Waiting time depends on the document.** The 4B's warm median per fragment was about 25 s in two examples and 42.7 s in `report vs
news`, which is over the 30 s preference. There the prompts were longer (about 1,050 tokens), none reused a cached prefix and most
generation ran at about 4 tokens/s rather than 7. The CPU speed limit was below 100 % in 13 of the 97 samples of that block and other
applications were open, so the cause is not established.

**The comment path found a real bug.** In the frozen run every Qwen3.5 comment came back empty, after 80 to 150 s per fragment. Qwen3.5
thinks by default, the adapter did not turn that off, and the 512 tokens allowed for a comment ran out before any answer. This says
nothing about the quality of those models: it is an integration failure, and the structured judge above, with `think: false`, did not
have it.

### Follow-up outside the protocol: comments with thinking off

Commit `66c9552` adds `think=False` to the adapter, after the frozen run and without changing any of its results. These rows come from a
separate run of the same three documents through the same API, with the app's prompts and options. That includes what Qwen3.5 inherits
from its model file, such as `presence_penalty` 1.5, because the app does not override it. Comments are free text with no label, so they
are not scored. Numbered-list markers are removed before the text checks.

<!-- BEGIN GENERATED: app_comments_followup -->
| Model | Mode | Non-empty comments | Whole request, s | Comment per fragment (first / warm median), s | Both values named | Comments with a figure absent from the inputs | Comments with a capitalised word absent from the inputs |
|---|---|---|---|---|---|---|---|
| 1B | text vs text | 4/4 | 52.9 | 17.1 / 11.7 | 1/2 | 0/4 | 0/4 |
| 1B | report vs news | 4/4 | 77.2 | 19.4 / 20.1 | 0/1 | 2/4 | 1/4 |
| 1B | news vs newsletter | 6/6 | 119.3 | 28.1 / 18.4 | 0/1 | 0/6 | 1/6 |
| 4B | text vs text | 4/4 | 101.6 | 28.3 / 22.5 | 2/2 | 1/4 | 0/4 |
| 4B | report vs news | 4/4 | 130.3 | 33.0 / 35.1 | 1/1 | 0/4 | 0/4 |
| 4B | news vs newsletter | 6/6 | 176.6 | 33.9 / 28.5 | 1/1 | 0/6 | 3/6 |
| 9B | text vs text | 4/4 | 181.2 | 55.1 / 41.4 | 2/2 | 0/4 | 0/4 |
| 9B | report vs news | 4/4 | 248.1 | 76.5 / 53.7 | 1/1 | 0/4 | 0/4 |
| 9B | news vs newsletter | 6/6 | 340.4 | 67.1 / 52.7 | 1/1 | 0/6 | 0/6 |
<!-- END GENERATED: app_comments_followup -->

**The fix works.** Every Qwen3.5 comment now arrives (14/14 per model, against 0/14 in the frozen run), and a whole request with the 4B
takes 102 to 177 s rather than 332 to 472 s. Its warm median per fragment is 22.5 to 35.1 s, so the 30 s preference is met on two
documents and missed on one. With the 9B it is 41.4 to 53.7 s.

**The comments look better than the 1B's, and are still not clean.** Read by the assistant, not by a human reviewer. On the four fragments
with values to check, both Qwen3.5 models named both values every time and the 1B once. Around that main point, though, they add remarks
about omissions that do not matter, and they can be wrong. On `text vs text` #3 the 4B says the fragment does not mention the exceptions
for residents and people with reduced mobility, when the fragment states exactly that.

The 1B's comments are word for word the same as in the frozen run (14/14), so the option changes nothing for it.

Two of the automatic flags are not invented data: the figure flagged for the 4B is a citation to `párr. 2` of a passage labelled
`párr. 1-4`, and its flagged capitalised word is `No` at the start of a bullet each time.

## The 1B with its inherited configuration

The common protocol changed the format, so an improvement or a decline could come from the prompt rather than from the model. Six
evaluation cases were therefore also run with the app's own prompt and settings. The output is free text, so there are no labels to score, and
these rows are not comparable with the tables above. Numbered-list markers are removed before the text checks, because otherwise a "3."
counts as the figure 3: that had made `EV-23` look as if it named the claim's 3 %.

<!-- BEGIN GENERATED: legacy -->
| Case | Expected | Status | Wall s | Output tokens | Both values named | Figures absent from the inputs | Capitalised words absent from the inputs |
|---|---|---|---|---|---|---|---|
| `EV-02` | `contradicted` | ok | 17.6 | 186 | yes | none | none |
| `EV-11` | `contradicted` | ok | 28.1 | 402 | yes | none | none |
| `EV-12` | `insufficient_evidence` | ok | 24.6 | 376 | n/a | none | none |
| `EV-17` | `contradicted` | ok | 20.6 | 308 | yes | none | none |
| `EV-23` | `contradicted` | ok | 26.2 | 378 | **no** | none | none |
| `EV-26` | `contradicted` | ok | 33.1 | 504 | yes | none | none |
<!-- END GENERATED: legacy -->

All six comments arrived in 17.6 to 33.1 s. Naming both values does not mean noticing the conflict. Read by the assistant, not by a
human reviewer:

- `EV-02` is the only one that presents the two values as a discrepancy (60 against 90 days).
- `EV-11` and `EV-26` state what each passage says without saying that the claim is wrong.
- `EV-17` gives both 35.000 euros and 35 millones as if they were compatible.
- `EV-12` says the report mentions the 30 days, which appear only in the claim.

The end-to-end example shows the same pattern. For the news item that says 60 days where the report says 90 (`report vs news`, #1), the
1B's comment mentions neither figure and states that the entity's population is 100.000 people, a number found in no input.

So the weak score in section A is probably not only an effect of the new format. These comments were not scored, though, so this is an
observation and not a measurement.

## Decision

The rule applied mechanically to the saved results:

<!-- BEGIN GENERATED: selection -->
- Complete candidates in both passes: 1B, 4B, 9B.
- Order by false supports, then macro-F1, in pass 1: 9B > 4B > 1B.
- Order changes in the other pass: yes.
- Robust winner under the rule: none.
- Median warm time within the 30 s preference: 1B yes, 4B yes, 9B no.
- Critical false supports: 1B 13, 4B 1, 9B 0.
- Comments stay off by default under the rule: yes.
<!-- END GENERATED: selection -->

**Recommended if comments are turned on: `qwen3.5:4b-q4_K_M`.** The rule names no robust winner: the 9B ranks first in pass 1 (0 false
supports against 1) and the 4B in pass 2 (both 0, macro-F1 1.00 against 0.95). It then prefers the lower cost. The 4B took a median of
21.8 s against 38.8 s, and its weights take 2.65 GB against 5.63 GB.

**Why not the 9B.** Its advantage is one case, `EV-24`, which the 4B got right in pass 2 and the 9B got wrong in pass 1 with a different
label. It also missed two cases in both passes that the 4B got right (`EV-04` and `EV-27`), and it is over the 30 s preference. If a
false support matters far more to you than the wait, choosing the 9B is defensible, but this run does not show that it is safer. Separate
quality and speed profiles would add nothing concrete, so none is offered.

**Why not the 1B.** It is the fastest and lightest, and the gap is not close: 13 of 18 critical cases falsely supported, against 1 and 0.

**Comments stay off by default and experimental.** The 4B's critical false support on `EV-24` is unresolved. It disappeared in pass 2 with
the same prompt, which is not a fix.

**The configured model stays `llama3.2:1b` for now.** The evidence for the 4B comes from the structured judge, which the app does not use.
The comments the app shows have not been scored, and the follow-up shows that they can be wrong. Changing the default is one line in
`IA-modulable/config/config.yaml` and a larger download, so it is left as an explicit decision for the project's owner, after a person has
reviewed these comments. Until then, use `LLM_MODEL=qwen3.5:4b-q4_K_M`.

**One change to the application.** The adapter now sends `think=False`, because without it no Qwen3.5 model can produce a comment through
`LLM_MODEL`. Comments remain off by default and the configured model is the same.

**Still pending:** a person reviewing the labels and the explanations, scoring the app's comments, and trying cases from real news.

## Limitations

- **Small and synthetic.** 30 cases written by the assistant, with no human review of cases or labels. Claims are atomic and tidy, the
  evidence is explicit, and real news will be messier. Even a perfect score here would not establish general reliability, and
  small differences are not statistically significant. The confidence intervals treat cases as independent, which they are not, so
  read them as a rough guide.
- **Not a neutral choice of cases.** The set covers kinds of error already seen with the 1B, such as the 60/90-day deadline, which may
  tilt it against that model.
- **Correlated cases.** Three claims per scenario, and two passes at temperature 0 are not independent samples.
- **Explanations not semantically reviewed.** Automatic checks cover IDs, quotes and figures only. Human review is pending.
- **Single machine, single run.** Timings depend on this laptop, its temperature and whatever else was running. The other applications
  were left open, and thermal throttling occurred: the CPU speed limit is reported per block.
- **Model plus configuration.** These results describe each model with this prompt, these parameters and thinking off. They do not
  isolate the effect of size, and thinking-on results were not evaluated in part A.
- **End-to-end is an integration check.** One example per mode, with few fragments. It is not a retrieval evaluation, and it does not
  extrapolate to long documents.
- **Multi-claim paragraphs.** One label per fragment cannot describe a paragraph with several claims, as the mixed paragraph in the
  `text_vs_text` example shows. By convention it counts as `contradicted` if any claim is, so `supported` there would hide a problem.
- **Not a product verdict.** The app still does not issue labels. They exist only in this benchmark, to make the models' behaviour scorable.
  So the quality figures describe the structured judge and not the free-text comments that the app shows.

## Reproduce

Commands, how to resume an interrupted run and how this document was generated are in [`benchmarks/README.md`](../benchmarks/README.md).
