#!/usr/bin/env python3
"""Public download/secure archive extraction/quality control and source-only split manifests."""
import _bootstrap  # noqa
import argparse
from pathlib import Path
from mangonet.config import ALL_DATASETS
from mangonet.data.download import download_public_dataset, extract_archives
from mangonet.data.quality_control import build_manifest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root', default='data', help='Root containing raw/<dataset>/<class>/** images')
p.add_argument('--download', nargs='*', choices=list(ALL_DATASETS), default=[], help='Download selected public Mendeley archives/files')
p.add_argument('--extract', action='store_true', help='Safely extract archives under raw/<dataset>/')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--min-side', type=int, default=64)
p.add_argument('--hamming', type=int, default=6)
p.add_argument('--exclude-file', default=None, help='Newline-delimited excluded relative image paths')
p.add_argument('--label-overrides-csv', default=None, help='path,label CSV for unrecognised source folder names')
p.add_argument('--confirmed-pairs-csv', default=None, help='a,b,decision CSV from manual duplicate verification')
p.add_argument('--no-qc', action='store_true')
args = p.parse_args()
root = Path(args.data_root)
for name in args.download:
    print(f'Downloading {name} ...', flush=True)
    try:
        print('Downloaded:', download_public_dataset(name, root/'raw'/name))
    except Exception as exc:
        print(f'[ACTION REQUIRED] {name}: {exc}; download manually from its Mendeley Data page and extract into raw/{name}.')
        raise SystemExit(1)
if args.extract:
    for name in ALL_DATASETS:
        folder = root/'raw'/name
        if folder.exists():
            print('Extracted:', extract_archives(folder))
if not args.no_qc:
    frame = build_manifest(root, seed=args.seed, min_side=args.min_side, hamming=args.hamming,
                           exclude_file=args.exclude_file, label_overrides_csv=args.label_overrides_csv,
                           confirmed_pairs_csv=args.confirmed_pairs_csv)
    print(f'Wrote {len(frame)} validated records to {root}/processed/manifest.csv')
    print(frame.groupby(['dataset','split']).size().to_string())
    print('Review duplicate_review.csv, rejected.csv, qc_summary.json before final experiments.')
