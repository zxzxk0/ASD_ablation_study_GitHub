# v5 repository packaging status

Included: original repository, supplied additions, latest separately uploaded gaze/image/audit scripts, S5 ridge and identity/metadata analyses, E1-E4 replay, MetaCA-MIL stopping-rule comparison, leakage figure generator, literature-audit evidence and builder.

Still missing from the supplied materials:
- Claude Opus 4.5 implementation, prompts/settings as applicable, and saved results.
- Final saved outputs for MetaCA-MIL, E1-E4, S5, and canonical gaze (including the confirmed B=2000 family-selection result).
- Exact environment versions and run configurations for the reported experiments.
- Author-selected software license.

The MetaCA-MIL README command is an explicit suggested invocation, not a verified historical command. Bootstrap count and all other options must match the saved run configuration. No outputs or execution provenance were invented. Manuscript numeric agreement has not been checked against final CSVs because those outputs were not supplied.

Original computational code was retained. The canonical gaze script's docstring example was updated from 200 to 2000 nested-family permutations; no algorithm changes were made. Uploaded filenames were normalized by removing attachment copy suffixes. Historical plotting code is in legacy/.

Validation: Python AST parsing for all source files; both S5 synthetic self-tests passed; ZIP integrity passed. A nearly constant-input correlation warning occurred in the ridge synthetic test; it did not fail the assertions. Full training, original-data analyses, live LLM calls, and every figure were not executed.
