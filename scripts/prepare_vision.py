"""Prepare hash-verified sources/checkpoints for the offline CPU release image."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen
from zipfile import ZipFile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(asset, cache):
    target = cache / asset['sha256']
    if target.is_file() and digest(target) == asset['sha256']:
        return target
    for attempt in range(3):
        try:
            with urlopen(asset['url'], timeout=120) as source, target.with_suffix('.part').open('wb') as dest:
                shutil.copyfileobj(source, dest)
            if digest(target.with_suffix('.part')) != asset['sha256']:
                raise ValueError('Checksum mismatch for '+asset['url'])
            target.with_suffix('.part').replace(target)
            return target
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)


def prepare(manifest, hub, models, cache):
    data = json.loads(manifest.read_text())
    for directory in [hub, models, cache, hub/'checkpoints']:
        directory.mkdir(parents=True, exist_ok=True)
    locations = {}
    for name, asset in data['sources'].items():
        archive = fetch(asset, cache)
        directory = hub/(asset['repository'].replace('/', '_')+'_'+asset['revision'])
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            with ZipFile(archive) as z:
                for member in z.infolist():
                    relative = Path(member.filename)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Unsafe source archive path')
                z.extractall(tmp)
            roots = list(Path(tmp).iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError('Unexpected source archive layout')
            if directory.exists():
                shutil.rmtree(directory)
            shutil.copytree(roots[0], directory)
        locations[name] = directory
    # Upstream Gaze-LLE hardcodes an unversioned DINOv2 Hub reference. Apply only
    # this verified, explicit source change in the prepared image's private cache.
    backbone = locations['gazelle']/'gazelle/backbone.py'
    original = backbone.read_text()
    needle = "torch.hub.load('facebookresearch/dinov2', model_name)"
    if original.count(needle) != 1:
        raise ValueError('Upstream backbone changed; review the DINOv2 pin patch')
    pinned = "torch.hub.load('facebookresearch/dinov2:"+data['sources']['dinov2']['revision']+"', model_name, trust_repo=True)"
    backbone.write_text(original.replace(needle, pinned))
    for asset in data['weights']:
        destination = (hub/'checkpoints' if asset['name'].endswith('.pth') else models)/asset['name']
        shutil.copyfile(fetch(asset, cache), destination)
    (hub/'trusted_list').write_text('fkryan_gazelle\nfacebookresearch_dinov2\n')
    data['patch'] = {'file': 'gazelle/backbone.py', 'description': 'Pin nested DINOv2 Hub load',
                     'patched_sha256': digest(backbone)}
    (models/'vision-assets-prepared.json').write_text(json.dumps(data, indent=2)+'\n')
    print('Verified sources and four checkpoints prepared for offline inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('configs/vision-assets.json'))
    parser.add_argument('--hub', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.hub, args.models, args.cache)
