# Note on the commit history

On 5 November 2026, after every evaluation in this repository had run, the start of the history was reorganised. The first two commits, a
clean copy of the thesis prototype and its adaptation, were combined into one initial commit. Its files are exactly those of the adaptation,
so this is still [an adapted prototype](../README.md#how-this-was-built), not something developed from scratch.

Every later commit keeps its files, message, author and dates, and has a new identifier because its ancestors changed. The merge of the first
pull request is no longer signed by GitHub, since a signature cannot carry over to a new commit. The project's final files did not change.

## What this means for the evaluations

Nothing was rerun or edited. The evaluations ran from the original commits, and those are the identifiers in the manifests, saved results,
generated tables, reports and commit messages. They are the record of what ran, so they stay as they are.

The equivalent commits below were created during the reorganisation, after those runs. They let you check out the same files today, but they do not
show what was frozen or pushed before a run: that statement is about the original commits. The dates shown on them are the original ones,
not the day they were recreated.

## Equivalent commits

The original commits are no longer on this repository's branches. To reproduce an evaluation, use its equivalent.

| What ran or changed | Original commit | Current equivalent | Git tree, the same in both |
|---|---|---|---|
| Protocol freeze and the frozen comparison, `2026-11-05_protocol-v1` | `1af70980aa31` | `c043e96db348` | `e8bb3fb7635f` |
| The `think=False` change cited in the comparison report | `66c955297d21` | `0a25ad8c1397` | `9c7db32e4f4c` |
| Comments follow-up, `2026-11-05_app-comments-followup` | `b52eb986c45f` | `6604f7a3cdce` | `0f70ec10d926` |
| Integration check v1, `2026-11-05_integration-check-v1` | `92200b5ba715` | `d1f157d6fc3f` | `7f6568f2cdfa` |
| Integration check v2, `2026-11-05_integration-check-v2` | `9892f6ef4f29` | `ab2e27be779a` | `e7b100285dc8` |

Here, equivalent means that both commits have the same Git tree, and so the same files, byte for byte. This was checked for every later commit when the
history was reorganised. To see the tree of a current commit:

```bash
git rev-parse c043e96^{tree}
```

[`benchmarks/README.md`](../benchmarks/README.md) explains how to run them from a separate worktree.
