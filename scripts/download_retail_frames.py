"""Download original every-fourth-frame JPEGs for selected Retail Gaze sessions."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--sessions', nargs='+', required=True)
parser.add_argument('--frame-count', type=int, required=True, help='Verified original image count per selected session')
args = parser.parse_args()
if args.frame_count < 1:
    parser.error('Frame count must be positive')
manifest = []
for session in args.sessions:
    subject, timestamp = session.split('_')
    if not subject.isdigit() or not timestamp.isdigit():
        parser.error('Sessions must be numeric subject_timestamp pairs')
    for number in range(0, args.frame_count*4, 4):
        relative = f'{subject}/{timestamp}/{number}.jpg'
        path = args.output/relative
        url = ('https://www.kaggle.com/api/v1/datasets/download/dulanim/retailgaze/'
               + quote('RetailGaze_V2_seg/RetailGaze_V2/'+relative, safe=''))
        if not path.exists():
            with urlopen(url, timeout=120) as response:
                data = response.read()
            if not data.startswith(b'\xff\xd8'):
                raise ValueError(f'Expected JPEG: {relative}')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.replace(path)
        data = path.read_bytes()
        if not data.startswith(b'\xff\xd8'):
            raise ValueError(f'Invalid JPEG: {relative}')
        manifest.append({'path': relative, 'url': url, 'sha256': hashlib.sha256(data).hexdigest()})
    print(session, flush=True)
(args.output/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
