# Roadmap

ShelfSense is a research baseline. These are priorities, not delivery promises.

## Current evidence

The first three-session Retail Gaze diagnostic scored 43 test labels. Neither
annotated-head nor automatic-head inference hit any of the 31 eligible target
regions. The pipeline and visual exports work; precise product attention does not.
See [the baseline](docs/retail-gaze.md#initial-measured-baseline).

## Evaluation first

- Audit gaze-point/mask inconsistencies and document correction criteria.
- Expand evaluation across camera angles and participants; use participant/session
  separation when training. Keep validation and test roles distinct.
- Compare models on identical accepted subsets as well as coverage and all-label metrics.
- Validate attention durations with original timestamps and independently labeled episodes.

## Model and tracking improvements

- Evaluate a dedicated head detector for rear views, masks and occlusions.
- Investigate gaze adaptation or alternatives using validation data; annotated-head
  failures show that head detection alone is not the complete problem.
- Measure tracking continuity with identity annotations and crowded scenes.
- Profile sustained CPU latency and memory on representative footage before optimization.

## Release engineering

- Keep package locks, model hashes and source revisions current through tested updates.
- Extend automated checks for video export and reproducible dataset preparation.
- Publish tagged research releases with changelogs and explicit known limitations.

A production claim requires measured deployment-camera accuracy, robust unknown
handling, duration validation and an appropriate security/privacy deployment design.
