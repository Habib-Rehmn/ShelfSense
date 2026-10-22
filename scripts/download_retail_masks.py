"""Fetch only product masks referenced by selected Retail Gaze sessions."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from shelfsense.retail_gaze import load_sessions

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', type=Path)
parser.add_argument('--sessions', nargs='+', required=True)
args = parser.parse_args()
sessions = load_sessions(args.dataset, args.sessions, 'all')
paths = sorted({label['mask'] for session in sessions.values()
                for label in session['labels'].values() if label['mask'] is not None})
manifest = []
for path in paths:
    relative = path.relative_to(args.dataset/'masks').as_posix()
    url = ('https://www.kaggle.com/api/v1/datasets/download/dulanim/retailgaze/'
           + quote('RetailGaze_V2_seg/RetailGaze_V2/'+relative, safe=''))
    if not path.exists():
        with urlopen(url, timeout=120) as response:
            data = response.read()
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError(f'Expected PNG, received another response for {relative}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.part')
        temporary.write_bytes(data)
        temporary.replace(path)
    data = path.read_bytes()
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError(f'Invalid PNG: {path}')
    manifest.append({'path': relative, 'url': url, 'sha256': hashlib.sha256(data).hexdigest()})
    print(relative)
(args.dataset/'masks'/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
