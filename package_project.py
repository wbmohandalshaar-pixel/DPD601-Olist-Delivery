"""Package tracked source AND ignored data/results, plus verifiable Git history.

Run after the full pipeline, tests and report render. Output must be outside
the project so an older ZIP cannot recursively become part of the delivery.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile
from src.data import require_raw_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Choose an output ZIP outside the project directory')
    require_raw_files(root / 'data/raw')
    required = ['data/processed/all_orders.csv', 'data/processed/model_orders.csv',
                'outputs/tables/test_metrics.csv', 'outputs/tables/selection.json',
                'outputs/models/selected_model.joblib', 'docs/Report_AR.docx',
                'docs/Report_AR.pdf']
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip():
        raise RuntimeError('Commit source/report changes before packaging')
    bundle = root / 'git_history.bundle'
    subprocess.run(['git', 'bundle', 'create', str(bundle), '--all'], cwd=root, check=True)
    subprocess.run(['git', 'bundle', 'verify', str(bundle)], cwd=root, check=True)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    files = {root / name for name in tracked if name}
    for folder in ['data/raw', 'data/processed', 'outputs']:
        files.update(p for p in (root / folder).rglob('*') if p.is_file()
                     and not any(part.startswith('.') for part in p.relative_to(root).parts)
                     and '__pycache__' not in p.parts and '.tmp' not in p.name)
    files.add(bundle)
    manifest = root / 'MANIFEST_SHA256.txt'
    manifest.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' +
                                p.relative_to(root).as_posix() + '\n'
                                for p in sorted(files)), encoding='utf-8')
    files.add(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, 'DPD601/' + path.relative_to(root).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise IOError('ZIP integrity check failed')
        for line in manifest.read_text().splitlines():
            digest, name = line.split('  ', 1)
            if hashlib.sha256(archive.read('DPD601/' + name)).hexdigest() != digest:
                raise IOError('Archive hash mismatch: ' + name)
    temporary.replace(output)
    print(f'Verified {len(files)} files: {output}')


if __name__ == '__main__':
    main()
