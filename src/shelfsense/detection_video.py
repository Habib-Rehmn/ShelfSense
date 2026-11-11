"""Run automatic person/head/gaze inference on every frame and export a demo."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
from time import perf_counter

from .gazelle import SceneGazeAnalyzer
from .gazelle_adapters import GazelleEstimator, PoseHeadDetector
from .provenance import environment_manifest
from .retail_evaluation import OriginalFrames
from .retail_video import draw_target, panel
from .tracking import YoloByteTracker


def run(args):
    import cv2
    import numpy as np
    import torch

    if not isfinite(args.fps) or args.fps <= 0 or args.threads < 1:
        raise ValueError('FPS and CPU thread count must be positive')
    if args.output_dir.exists():
        raise ValueError('Choose a new output directory')
    source = OriginalFrames(args.frames)
    for path in [args.weights, args.head_weights, args.gaze_weights]:
        if not path.is_file():
            raise ValueError(f'Missing checkpoint: {path}')
    torch.set_num_threads(args.threads)
    started = perf_counter()
    tracker = YoloByteTracker(str(args.weights), device=args.device)
    gaze = SceneGazeAnalyzer(PoseHeadDetector(str(args.head_weights), args.device),
                            GazelleEstimator(str(args.gaze_weights), args.device))
    args.output_dir.mkdir(parents=True)
    writer, count, person_frames, gaze_frames = None, 0, 0, 0
    reasons, ids = Counter(), set()
    person_color, gaze_color = (255, 190, 70), (70, 230, 255)
    try:
        with (args.output_dir/'predictions.jsonl').open('w') as output:
            while True:
                ok, image = source.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                tracks = tracker.track(image)
                observations = gaze.observe(image, tracks)
                record = {'frame_index': count, 'original_image': source.paths[count].name,
                          'playback_timestamp_s': count/args.fps, 'image_size': [width, height],
                          'tracks': [asdict(t) for t in tracks],
                          'gaze': [asdict(o) for o in observations]}
                output.write(json.dumps(record, allow_nan=False)+'\n')
                output.flush()
                person_frames += bool(tracks)
                accepted = sum(o.target_image is not None for o in observations)
                gaze_frames += accepted > 0
                canvas = image.copy()
                by_id = {o.track_id: o for o in observations}
                for track in tracks:
                    ids.add(track.track_id)
                    box = tuple(round(v) for v in track.bbox)
                    cv2.rectangle(canvas, box[:2], box[2:], person_color, 2)
                    label = f'Person #{track.track_id} | det {track.confidence:.2f}'
                    x, y = max(0, min(box[0], width-250)), max(18, box[1]-8)
                    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, person_color, 1, cv2.LINE_AA)
                    observation = by_id[track.track_id]
                    draw_target(cv2, canvas, observation.head_bbox, observation.target_image, gaze_color)
                    if observation.target_image is None:
                        reasons[observation.reason] += 1
                status = (f'{len(tracks)} tracked | {accepted} gaze estimate(s)' if tracks else 'No tracked person')
                unavailable = sorted({o.reason.replace('_', ' ') for o in observations if o.reason})
                if unavailable:
                    status += ' | Gaze unavailable: '+', '.join(unavailable)
                canvas = panel(cv2, canvas, 'ShelfSense | Automatic detection + gaze',
                               f'Frame {count+1}/{len(source.paths)} | '+status, person_color)
                footer = np.zeros((52, width, 3), dtype=image.dtype)
                footer[:] = (28, 24, 20)
                cv2.putText(footer, 'Blue: person + track ID   Yellow: head + predicted gaze', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .46, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.putText(footer, 'Research demo | Every frame inferred | Playback is not inference speed', (10, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, (190, 190, 190), 1, cv2.LINE_AA)
                rendered = np.concatenate([canvas, footer], axis=0)
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output_dir/'detection.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                             (rendered.shape[1], rendered.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot open video writer')
                writer.write(rendered)
                if count in (0, 26, 98):
                    if not cv2.imwrite(str(args.output_dir/f'preview-{count:03d}.jpg'), rendered):
                        raise OSError('Could not write preview')
                count += 1
                if count % 20 == 0:
                    print(f'Processed {count}/{len(source.paths)} frames', flush=True)
    finally:
        source.release()
        if writer is not None:
            writer.release()
    if count != len(source.paths):
        raise ValueError('Incomplete image sequence')
    report = {'frames': count, 'frames_with_tracks': person_frames,
              'frames_with_gaze': gaze_frames, 'track_ids': sorted(ids),
              'unknown_observations': dict(reasons), 'elapsed_s': perf_counter()-started,
              'playback_fps': args.fps, 'playback_duration_s': count/args.fps,
              'inference_on_every_frame': True, 'annotations_used_for_inference': False,
              'accuracy_evaluation': False,
              'notes': ['No dataset labels, masks or ground-truth head boxes are used.',
                        'Gaze arrows are predictions, not verified product attention.',
                        'Arrow origin is a heuristic point within the detected head box.',
                        'Playback timing is illustrative, not measured inference FPS.']}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    inputs = [args.weights, args.head_weights, args.gaze_weights, *source.paths]
    manifest = environment_manifest(inputs)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', type=Path, help='Original JPEG directory: 0.jpg, 4.jpg, ...')
    for name in ['weights', 'head-weights', 'gaze-weights', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Detection video failed: {exc}\n')


if __name__ == '__main__':
    main()
