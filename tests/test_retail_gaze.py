import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shelfsense.retail_gaze import load_sessions, match_track, prediction_record, summarize
from shelfsense.tracking import Track


class RetailTests(unittest.TestCase):
    def fixture(self, root, frame_number=1):
        (root/'samples.json').write_text(json.dumps({'samples': [
            {'_id': {'$oid': 'a'}, 'subject_id': 1, 'session_id': '001', 'filepath': 'data/a.mp4'},
            {'_id': {'$oid': 'b'}, 'subject_id': 2, 'session_id': '001', 'filepath': 'data/b.mp4'}]}))
        frames = [{'_sample_id': {'$oid': sid}, 'frame_number': frame_number,
                   'split': split, 'head': {'bounding_box': [.1, .2, .3, .4]},
                   'gaze_target': {'keypoints': [{'points': [[.5, .6]]}]},
                   'scene_mask': {'mask_path': f'/publisher/RetailGaze_V2/{subject}/001/masks/mask_out1.png'}}
                  for sid, subject, split in [('a', 1, 'test'), ('b', 2, 'train')]]
        (root/'frames.json').write_text(json.dumps({'frames': frames}))

    def test_alignment_conversion_and_local_mask(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            sessions = load_sessions(root, ['1_001'])
            label = sessions['1_001']['labels'][0]
            self.assertEqual(label['head'][:3], (.1, .2, .4))
            self.assertAlmostEqual(label['head'][3], .6)
            self.assertEqual(label['mask'], root/'masks/1/001/masks/mask_out1.png')
            self.assertEqual(set(sessions), {'1_001'})
            with self.assertRaises(ValueError): load_sessions(root, ['2_001'])

    def test_bad_frame_and_duplicate_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root, 0)
            with self.assertRaises(ValueError): load_sessions(root, ['1_001'])
            self.fixture(root)
            doc = json.loads((root/'frames.json').read_text())
            doc['frames'].append(doc['frames'][0])
            (root/'frames.json').write_text(json.dumps(doc))
            with self.assertRaises(ValueError): load_sessions(root, ['1_001'])

    def test_abstention_and_missing_mask_denominators(self):
        mask = [[False, False], [False, True]]
        hit = prediction_record((1., 1.), None, (1., 1.), (2, 2), mask, 'valid')
        unknown = prediction_record(None, 'no_head', (1., 1.), (2, 2), mask, 'valid')
        missing = prediction_record((0., 0.), None, (1., 1.), (2, 2))
        result = summarize([hit, unknown, missing])
        self.assertEqual(result['coverage'], 2/3)
        self.assertEqual(result['region_eligible'], 2)
        self.assertEqual(result['region_accuracy_all'], .5)
        self.assertEqual(result['region_accuracy_accepted'], 1.)
        self.assertAlmostEqual(result['mean_error_normalized_xy_accepted'], 2**.5/4)
        self.assertEqual(result['unknown_reasons'], {'no_head': 1})
        self.assertIsNone(summarize([unknown])['mean_error_px_accepted'])

    def test_matching_uses_person_location_and_rejects_ambiguity(self):
        a = Track(1, (0, 0, 100, 200), .9)
        b = Track(2, (200, 0, 300, 200), .9)
        self.assertEqual(match_track([a,b], (10, 10, 30, 30)), a)
        self.assertIsNone(match_track([a, Track(3, a.bbox, .8)], (10, 10, 30, 30)))
        self.assertIsNone(match_track([a], (10, 150, 30, 170)))

class OriginalFrameTests(unittest.TestCase):
    def test_numeric_sequence_order_and_gap_rejection(self):
        from shelfsense.retail_evaluation import OriginalFrames
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in [12, 0, 8, 4]: (root/f'{n}.jpg').touch()
            self.assertEqual([p.stem for p in OriginalFrames(root).paths], ['0', '4', '8', '12'])
            (root/'4.jpg').unlink()
            with self.assertRaises(ValueError): OriginalFrames(root)

    def test_mask_alpha_and_inconsistent_label(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('Requires vision extra')
        from shelfsense.retail_evaluation import load_mask
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'mask.png'
            image = np.full((2, 2, 4), 255, dtype=np.uint8)
            image[0, 0, 3] = 0
            cv2.imwrite(str(path), image)
            mask, status = load_mask(path, (2, 2), (1., 1.))
            self.assertEqual(status, 'valid')
            self.assertFalse(mask[0, 0])
            self.assertEqual(load_mask(path, (2, 2), (0., 0.))[1], 'label_outside_mask')
            with self.assertRaises(ValueError): load_mask(path, (3, 3), (1., 1.))


if __name__ == '__main__':
    unittest.main()
