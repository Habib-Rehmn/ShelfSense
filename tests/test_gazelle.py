from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from shelfsense.calibration import Calibration
from shelfsense.cli import main
from shelfsense.gazelle import ImageGazeObservation, SceneGazeAnalyzer, decode_heatmap
from shelfsense.gazelle_adapters import GazelleEstimator, PoseHeadDetector
from shelfsense.pipeline import Frame, process
from shelfsense.replay import map_gaze_file
from shelfsense.attention import AttentionEngine
from shelfsense.ingestion import read_export
from shelfsense.storage import Store
from shelfsense.tracking import Track

ROOT = Path(__file__).resolve().parents[1]


class Image:
    shape = (480, 640, 3)


class GazelleTests(unittest.TestCase):
    def calibration(self):
        return Calibration.load(ROOT / 'configs/shelf.image-target.example.json')

    def analyzer(self, heads=None):
        detector = SimpleNamespace(detect=MagicMock(return_value=heads if heads is not None else [(100, 50, 140, 90)]))
        estimator = SimpleNamespace(model_name='gazelle_dinov2_vitb14_inout', threshold=.5,
            predict=MagicMock(side_effect=lambda im, boxes: [((320, 240), .9, None) for _ in boxes]))
        return SceneGazeAnalyzer(detector, estimator)

    def test_heatmap_peak_uses_xy_cell_centers_and_out_of_frame_gate(self):
        self.assertEqual(decode_heatmap([[.1, .2], [.9, .3]], .8, (640, 480), .5), ((160, 360), None))
        self.assertEqual(decode_heatmap([[.1, .9]], .2, (640, 480), .5), (None, 'out_of_frame'))
        for heatmap in ([[.1, float('nan')]], [[.1], [.2, .3]], []):
            self.assertEqual(decode_heatmap(heatmap, .8, (640, 480), .5)[1], 'invalid_prediction')
        self.assertEqual(decode_heatmap([[.5, .5]], .8, (640, 480), .5)[1], 'uninformative_heatmap')

    def test_one_scene_call_for_multiple_heads_preserves_track_order(self):
        analyzer = self.analyzer([(100, 50, 140, 90), (400, 50, 440, 90)])
        tracks = [Track(8, (380, 0, 500, 400), .9), Track(2, (80, 0, 200, 400), .9)]
        observations = analyzer.observe(Image(), tracks)
        self.assertEqual([o.track_id for o in observations], [8, 2])
        self.assertTrue(all(o.status == 'estimated' for o in observations))
        self.assertEqual(analyzer.estimator.predict.call_count, 1)
        self.assertEqual(analyzer.estimator.predict.call_args.args[1], [(400, 50, 440, 90), (100, 50, 140, 90)])

    def test_empty_missing_ambiguous_and_small_heads_do_not_infer(self):
        track = Track(1, (80, 0, 200, 400), .9)
        for heads, tracks, reason in [([], [track], 'no_head'),
                ([(100, 50, 140, 90)], [track, Track(2, track.bbox, .9)], 'ambiguous_head'),
                ([(100, 50, 105, 55)], [track], 'head_too_small')]:
            analyzer = self.analyzer(heads)
            self.assertTrue(all(o.reason == reason for o in analyzer.observe(Image(), tracks)))
            analyzer.estimator.predict.assert_not_called()
        analyzer = self.analyzer()
        self.assertEqual(analyzer.observe(Image(), []), [])
        analyzer.detector.detect.assert_not_called()

    def test_pose_uses_visible_landmarks_and_rejects_body_detector(self):
        model = MagicMock(task='pose')
        keypoints = model.predict.return_value[0].keypoints
        keypoints.xy.cpu.return_value.tolist.return_value = [[[10, 20], [20, 20], [100, 100], [0, 0], [0, 0]]]
        keypoints.conf.cpu.return_value.tolist.return_value = [[.9, .9, .1, .1, .1]]
        with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda _: model)}):
            self.assertEqual(PoseHeadDetector('pose.pt').detect(Image()), [(5, 10, 25, 30)])
            model.task = 'detect'
            with self.assertRaises(ValueError):
                PoseHeadDetector('person.pt')

    def test_incomplete_decoder_checkpoint_fails_instead_of_random_parameters(self):
        torch, model = MagicMock(), MagicMock()
        model.state_dict.return_value = {'backbone.weight': 1, 'decoder.weight': 2}
        torch.load.return_value = {}
        torch.hub.load.return_value = (model, MagicMock())
        with patch.dict('sys.modules', {'torch': torch}):
            with self.assertRaisesRegex(ValueError, 'complete'):
                GazelleEstimator('wrong.pt')
        model.load_gazelle_state_dict.assert_not_called()
        torch.load.assert_called_once_with('wrong.pt', map_location='cpu', weights_only=True)

    def test_geometry_only_mapping_and_off_shelf(self):
        calibration = self.calibration()
        target = calibration.map_observation(ImageGazeObservation(1, 'estimated', target_image=(320, 240)), (640, 480))
        self.assertEqual(target.zone_id, 'zone_b')
        self.assertEqual(calibration.map_observation(ImageGazeObservation(1, 'estimated', target_image=(10, 10)), (640, 480)).reason, 'off_shelf')
        self.assertEqual(calibration.map_observation(ImageGazeObservation(1, 'estimated', target_image=(float('nan'), 10)), (640, 480)).reason, 'invalid_gaze')
        with self.assertRaises(ValueError):
            calibration.map_observation(ImageGazeObservation(1, 'unknown'), (320, 240))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'shelf.json'
            calibration.save(path)
            self.assertEqual(Calibration.load(path).summary(), calibration.summary())

    def test_replay_and_attention_export_ingest(self):
        analyzer = self.analyzer()
        tracker = SimpleNamespace(track=lambda _: [Track(1, (80, 0, 200, 400), .9)])
        frames = [Frame(i, i*.1, Image()) for i in range(11)]
        raw, mapped = io.StringIO(), io.StringIO()
        process(frames, tracker, raw, gaze=analyzer, session_id='scene')
        calibration = self.calibration()
        self.assertEqual(map_gaze_file(io.StringIO(raw.getvalue()), mapped, calibration), 11)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            map_gaze_file(io.StringIO(raw.getvalue()), io.StringIO(), Calibration.load(ROOT/'configs/shelf.synthetic.json'))
        output = io.StringIO()
        process(frames, tracker, output, gaze=analyzer, calibration=calibration,
                attention=AttentionEngine(calibration), session_id='scene')
        data = read_export(io.StringIO(output.getvalue()), calibration)
        with tempfile.TemporaryDirectory() as d:
            store = Store('sqlite:///' + str(Path(d)/'analytics.db'))
            store.initialize()
            self.assertTrue(store.ingest(data)['imported'])

    def test_cli_defaults_to_gazelle_and_requires_head_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('clip.avi', 'person.pt', 'gaze.pt', 'pose.pt'):
                (root/name).touch()
            args = ['track', str(root/'clip.avi'), '--weights', str(root/'person.pt'),
                    '--gaze-weights', str(root/'gaze.pt'), '--output', str(root/'out.jsonl')]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            video = MagicMock()
            video.__enter__.return_value = [Frame(0, 0, Image())]
            with patch('shelfsense.cli.VideoReader', return_value=video), \
                 patch('shelfsense.cli.YoloByteTracker') as tracker, \
                 patch('shelfsense.cli.PoseHeadDetector') as detector, \
                 patch('shelfsense.cli.GazelleEstimator') as estimator, redirect_stdout(io.StringIO()):
                tracker.return_value.track.return_value = []
                estimator.return_value.model_name = 'gazelle_dinov2_vitb14_inout'
                estimator.return_value.threshold = .5
                self.assertEqual(main(args + ['--head-weights', str(root/'pose.pt')]), 0)
                detector.assert_called_once_with(str(root/'pose.pt'), 'cpu')
                estimator.assert_called_once_with(str(root/'gaze.pt'), 'cpu', .5)
            self.assertEqual(json.loads((root/'out.jsonl').read_text())['gaze_kind'], 'image_target')
