"""Maintainer-only: extract the USB SDK core and license from the pinned wheel."""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True, type=Path)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'manifest.json').read_text())
    pinned = manifest['wheels'][0]
    if options.wheel.name != pinned['wheel'] or hashlib.sha256(options.wheel.read_bytes()).hexdigest() != pinned['sha256']:
        raise ValueError('Camera SDK wheel does not match manifest.json')
    metadata = f'pyrealsense2-{pinned["version"]}.dist-info/'
    retained = set(manifest['retained_runtime_files'])
    rows, hashes = [], {}
    with zipfile.ZipFile(options.wheel) as wheel:
        for member in wheel.namelist():
            if (member not in retained and member not in (metadata+'METADATA', metadata+'WHEEL')
                    and not member.startswith(metadata+'licenses/')) or member.endswith('/'):
                continue
            data = wheel.read(member)
            path = root/'python'/member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            digest = hashlib.sha256(data).digest()
            rows.append([member, 'sha256='+base64.urlsafe_b64encode(digest).decode().rstrip('='), len(data)])
            hashes[member] = digest.hex()
    with (root/'python'/(metadata+'RECORD')).open('w', newline='') as record:
        writer = csv.writer(record)
        writer.writerows(rows)
        writer.writerow([metadata+'RECORD', '', ''])
    manifest['installed_files_sha256'] = hashes
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Bundled USB SDK core:', root/'python')


if __name__ == '__main__':
    main()
