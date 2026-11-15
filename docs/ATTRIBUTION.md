# AI assistance and report appendix checklist

## Attribution for this version

ChatGPT was used for project planning. OpenAI Codex assisted with implementation
and troubleshooting, refactoring, experiment execution and validation, and
documentation. Anthropic Claude audited the Python code and README on
15 November 2026, with recommendations on reproducibility and documentation
(author-reported; exact model not recorded). Recraft generated the 20 images used in the
external evaluation.

The author remains responsible for the final scientific interpretation and for
checking the output of these tools against the recorded validation evidence.

## Available evidence and limits

| Contribution | Evidence available here | Limit / item for the author to confirm |
|---|---|---|
| ChatGPT planning | Author-provided attribution and task history | Original planning transcript, date, model/version and decisions adopted are not tracked. |
| Codex implementation and troubleshooting | Source code, refactoring and validation notes, clean-environment checks and isolated execution logs retained locally | Exact model/build and the complete original dialogue are not tracked. A Git author field alone cannot separate human and AI work. |
| Codex complementary analyses and validation | CNN augmentation, JPEG intervention, PCA, SD and end-to-end notes; saved CSV/JSON records and current execution traces | Execution records support what ran, not independent confirmation of every historical authorship claim. The author must review the final scientific interpretation. |
| Anthropic Claude code/README audit | The author identifies a 15 November 2026 audit of `code_for_review_v2.txt`, recorded outside the repository in `Markdown collé(4).md`, titled “Audit de code_for_review_v2.txt”; follow-up titled “Audit de la version finale”, also pasted into the ChatGPT conversation | Exact Claude model was not recorded. These external texts were identified by the author but were not directly inspected during this pass. They are not repository files. Map particular findings to adopted changes from the original excerpts. |
| Recraft image generation | `data/recraft/generation_log.csv`, 20 PNGs, `SHA256SUMS.txt`, and prompts/API code in `analysis/generate_recraft.py` | Log records model `recraftv4_1_flash_v2`, requested 1024x1024 size and UTC timestamps. It is a project log, not provider-signed provenance or a rights certificate. |

## Where precise attribution belongs in the report

- Methods/code appendix: identify AI-assisted implementation, refactoring and
  reproduction documentation; distinguish assistance from the
  author's choices of research question, splits, thresholds and interpretation.
  Confirm those human decisions against the actual working record.
- External evaluation: label the 20 Recraft artworks as synthetic evaluation
  inputs, include generator/model, ten prompts with two images each, dates,
  selection procedure, RGB/LANCZOS preprocessing and the small sample limitation.
  Cite the generation log rather than implying an independent external dataset.
- Exploratory/complementary analyses: identify Codex assistance in implementing,
  running and checking augmentation, JPEG intervention, PCA and source
  generalization. The report author remains responsible for accepting the
  conclusions, including descriptive versus inferential distinctions.
- Audit appendix: attribute specific Claude findings only after the original
  external audit excerpts are available for precise citation; do not present every
  subsequent fix as independently audited by Claude. This submission pass was performed with Codex.

## Items to retain in the appendix

1. Tool/provider, model or version when known, dates, purpose and scope.
2. Representative original prompts or transcript excerpts, marking summaries as
   summaries; do not invent missing prompts or identify a model from a brand alone.
3. A mapping from adopted suggestions to code paths, document sections and
   validation evidence, plus significant rejected or modified suggestions.
4. Which analyses and figures were AI-assisted, which checks were actually run,
   and which conclusions were reviewed and accepted by the author.
5. Recraft's original prompts and generation record without rewriting them;
   verify the usage terms applicable at generation separately if required.
6. Dataset citation, unresolved archive version/checksum and archive-specific
   license; historical MPS training provenance and unavailable exact hardware.

The external audit texts were identified by the author but not inspected here.
The course workspace was not supplied; this checklist does not certify compliance
with unseen course rules.
No new Recraft generation, external audit or retrospective transcript recovery
was performed. No personal drafts, credentials or private conversations are added
to the repository.
