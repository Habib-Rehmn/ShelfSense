"""Convert a reviewed pip installation report into a platform-specific wheel lock."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
report = json.loads(args.report.read_text())
lines = ['# Linux x86_64 / CPython 3.12 CPU wheel lock; other platforms need a separate lock.',
         '# Generated from a reviewed pip --report; validated by Dockerfile.vision.',
         '--require-hashes', '--only-binary=:all:']
for entry in sorted(report['install'], key=lambda x: x['metadata']['name'].lower()):
    name, version = entry['metadata']['name'], entry['metadata']['version']
    download = entry['download_info']
    url = download['url']
    if urlparse(url).scheme != 'https' or urlparse(url).hostname not in {
        'files.pythonhosted.org', 'download.pytorch.org', 'download-r2.pytorch.org'}:
        raise ValueError('Review unexpected artifact origin before publishing the lock')
    sha = download['archive_info']['hashes']['sha256']
    lines += [f'# {name}=={version}', f'{name} @ {url} --hash=sha256:{sha}']
args.output.write_text('\n'.join(lines)+'\n')
