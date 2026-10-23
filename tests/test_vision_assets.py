import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/prepare_vision.py'
spec = importlib.util.spec_from_file_location('prepare_vision', SCRIPT)
prepare_vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_vision)


class VisionAssetTests(unittest.TestCase):
    def test_verified_cached_sources_are_pinned_and_weights_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root/'cache'; cache.mkdir()
            sources = {}
            for name, repo in [('gazelle', 'fkryan/gazelle'), ('dinov2', 'facebookresearch/dinov2')]:
                archive = root/(name+'.zip')
                with ZipFile(archive, 'w') as z:
                    z.writestr(name+'/LICENSE', 'upstream notice')
                    if name == 'gazelle':
                        z.writestr(name+'/gazelle/backbone.py', "model = torch.hub.load('facebookresearch/dinov2', model_name)\n")
                sha = prepare_vision.digest(archive)
                archive.rename(cache/sha)
                sources[name] = {'sha256': sha, 'url': 'https://example.invalid/source', 'repository': repo, 'revision': name+'123'}
            data = b'test weight'; sha = hashlib.sha256(data).hexdigest(); (cache/sha).write_bytes(data)
            manifest = root/'assets.json'
            manifest.write_text(json.dumps({'sources': sources, 'weights': [{'name': 'test.pt', 'sha256': sha, 'url': 'https://example.invalid/weight'}]}))
            with patch.object(prepare_vision, 'urlopen', side_effect=AssertionError('Network should not be used')):
                prepare_vision.prepare(manifest, root/'hub', root/'models', cache)
            patched = root/'hub/fkryan_gazelle_gazelle123/gazelle/backbone.py'
            self.assertIn('facebookresearch/dinov2:dinov2123', patched.read_text())
            self.assertEqual((root/'models/test.pt').read_bytes(), data)
            self.assertTrue((root/'hub/fkryan_gazelle_gazelle123/LICENSE').is_file())
            result = json.loads((root/'models/vision-assets-prepared.json').read_text())
            self.assertEqual(result['patch']['patched_sha256'], prepare_vision.digest(patched))

    def test_download_checksum_mismatch_is_rejected(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            asset = {'url': 'https://example.invalid/file', 'sha256': '0'*64}
            with patch.object(prepare_vision, 'urlopen', return_value=io.BytesIO(b'wrong')):
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    prepare_vision.fetch(asset, Path(tmp))
            self.assertFalse((Path(tmp)/asset['sha256']).exists())

    def test_repository_manifest_pins_sha256_and_source_revisions(self):
        data = json.loads((SCRIPT.parents[1]/'configs/vision-assets.json').read_text())
        for source in data['sources'].values():
            self.assertRegex(source['revision'], r'^[a-f0-9]{40}$')
        for asset in [*data['sources'].values(), *data['weights']]:
            self.assertRegex(asset['sha256'], r'^[a-f0-9]{64}$')
            self.assertTrue(asset['url'].startswith('https://'))
