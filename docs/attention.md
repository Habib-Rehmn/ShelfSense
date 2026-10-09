# Module 4: temporal attention and events

ShelfSense now smooths shelf coordinates and derives **estimated attention
episodes** from repeated observations. These events do not establish physiological
eye fixations, interest, or purchases. All durations use video time, independent
of inference speed. Thresholds are configurable engineering defaults, not
scientifically validated values.

## Run the synthetic example

No vision dependencies or model downloads are needed:

```bash
PYTHONPATH=src python3 -m shelfsense map-gaze examples/attention-gaze.synthetic.jsonl \
  --calibration configs/shelf.synthetic.json --output outputs/module4-targets.jsonl
PYTHONPATH=src python3 -m shelfsense attention outputs/module4-targets.jsonl \
  --calibration configs/shelf.synthetic.json --output outputs/module4-attention.jsonl
```

This produces 23 frame records plus one session-ending record. The two completed
episodes are zone A for 0.8 seconds and zone C for 1.0 second. An unknown gaze
observation separates them; the shopper then disappears. These are synthetic
pipeline checks, not measured real-world results. Use new output paths to rerun.

For inference and temporal processing in a single pass:

```bash
shelfsense track data/shelf.mp4 --weights models/yolo26n.pt \
  --gaze-backend l2cs --gaze-weights models/L2CSNet_gaze360.pkl \
  --calibration outputs/calibration.json --attention \
  --output outputs/attention.jsonl
```

Alternatively, run `attention` on any schema-3 calibrated JSONL file. Both paths
use the same engine. The calibration ID and image resolution must match, with
strictly increasing timestamps and frame indices within each session. Inputs may
interleave sessions; state and track IDs remain isolated by session.

## Configuration

The following flags work on `attention` and `track`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--smoothing-tau-s` | 0.15 | EMA time constant, in seconds |
| `--min-duration-s` | 0.5 | Minimum candidate duration before confirming attention |
| `--max-gap-s` | 0.25 | Largest interval between valid observations allowed within an episode |
| `--max-radius` | 0.08 | Maximum Euclidean distance from the candidate's initial smoothed point |
| `--track-timeout-s` | 1.0 | Time since a track's last appearance before emitting shopper exit |

All values must be finite and positive; track timeout must be at least the
observation gap. Radius is in normalized shelf units: x is shelf width and y is
shelf height, so this is not a physical distance on a rectangular shelf.

The EMA weight is `1 - exp(-dt / tau)`. It smooths coordinates before zone
lookup, using elapsed media time rather than a fixed number of frames. Smoothing
can delay a zone transition; raw shelf predictions are retained for comparison.

An episode candidate starts at its first valid smoothed observation. Subsequent
points must stay in that zone and within the configured radius of its initial
point. This limits drift; it is not a full velocity/dispersion eye-fixation model.
Once minimum duration is reached, one start event is emitted. Shorter candidates
produce no attention start/end events.

## Gaps, exits and session endings

- A missing/unknown gaze estimate immediately closes attention at the previous
  valid observation and clears smoothing. It does not imply the shopper left.
- A missing track immediately closes attention and clears smoothing. Shopper
  exit waits until the timeout, or the end of the session.
- An interval longer than the gap limit closes the old episode before using the
  new observation. Its duration excludes the unobserved interval.
- A changed smoothed zone or excessive point movement ends the old episode at
  the previous observation and starts a new candidate at the current one.
- A point smoothed into a gap between zones becomes unknown. Off-shelf raw
  predictions are unknown before smoothing and never count as attention.
- A track returning after its timeout emits exit followed by entry, even if the
  tracker reused its number. Track IDs still do not establish human identity.
- Normal end-of-input closes all confirmed episodes at their last valid time and
  emits remaining shopper exits. No extra frame duration is guessed. Truncation
  with `--max-frames` is also end-of-input for this run.

An episode from timestamps 0.0 through 0.5 lasts 0.5 seconds, not six frame
intervals. Valid samples within the gap limit are assumed continuous between
observations; the system cannot observe what happened between frames.

## Schema 4 and event times

Frame records preserve schema-3 tracking, gaze and shelf targets, and add:

| Field | Meaning |
| --- | --- |
| `record_type` | `frame` |
| `attention_metadata` | Run ID, method, thresholds and calibration ID |
| `attention` | Smoothed point, zone, candidate duration and confirmed episode ID per track |
| `events` | Events emitted while processing this frame |

An estimated attention observation with a null `episode_id` is an unconfirmed
candidate, not a counted episode. Unknown observations have no episode ID.

Event types are `SHOPPER_ENTERED`, `SHOPPER_LEFT`, `ATTENTION_STARTED` and
`ATTENTION_ENDED`. Shopper entry/exit refers to video-track presence, not a
physical store entrance or shelf proximity. Events carry session ID, track ID,
shelf ID and a run-scoped unique event ID. Start/end events share `episode_id`
and `zone_id`; end events include start, end, duration and closure reason.

`timestamp_s` is the effective event time. `emitted_at_s` is when enough media
observations were available to emit it. For a 0.5-second threshold, an episode
starting at 0.0 is confirmed at 0.5: these fields are 0.0 and 0.5 respectively.
Exit events likewise refer back to the last observed track time. Event emission
order does not guarantee increasing effective event timestamps.

Successful completion adds one `record_type: session_end` record per session,
containing final events, timestamp and metadata, without a frame/attention list.
Consumers must check record type. A failed run may leave partial output without
these completion records; it must not be interpreted as a completed session.
Each invocation gets a new run ID. Replaying the same clip creates another run;
later analytics must select a run instead of summing repeated processing.

## Validation and remaining work

Tests cover timestamp-based EMA, confirmation timing, zone transitions,
within-zone movement, unknowns, dropped frames, track timeouts, session isolation,
calibration mismatches, malformed input, normal completion and CLI replay. The
example above exercises mapping through event output without mocked temporal
logic. Threshold tuning and held-out evaluation on real footage remain
outstanding. Database ingestion, API metrics and dashboard are the next module.
