"""Prepare a small create-only local Cloud Build context; no network or training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from citypulse_ai.api.station_runtime import ArtifactStore

ARTIFACTS = ('preprocessing.json', 'xgboost/model.ubj', 'xgboost/preprocessing.json')


def prepare(source, destination, repository):
    source, destination, repository = Path(source), Path(destination), Path(repository)
    if destination.exists():
        raise FileExistsError('Use a new context directory; existing files are never replaced')
    store = ArtifactStore(source)
    # Verify every source before creating output; store also validates the vocabulary.
    paths = {name: store.file(name) for name in ARTIFACTS}
    destination.mkdir(parents=True, exist_ok=False)
    for name, path in paths.items():
        target = destination/'artifacts'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        store_expected = store.specification['artifacts'][name]['sha256']
        if hashlib.sha256(target.read_bytes()).hexdigest() != store_expected:
            raise ValueError('Artifact changed during local staging; do not submit this context')
    # Reuse source unchanged, omit unrelated application components and all caches.
    package = repository/'src/citypulse_ai'
    target = destination/'src/citypulse_ai'
    target.mkdir(parents=True)
    shutil.copyfile(package/'__init__.py', target/'__init__.py')
    for name in ('api', 'features', 'forecasting', 'etl'):
        (target/name).mkdir()
        for path in (package/name).iterdir():
            approved = path.suffix == '.py' or (name == 'api' and path.name == 'station_artifacts.json')
            if path.is_file() and approved:
                if path.is_symlink():
                    raise ValueError('Source symlinks are not allowed in the build context')
                shutil.copyfile(path, target/name/path.name)
    for name in ('cloudrun-xgboost.Dockerfile', 'cloudrun-xgboost.Dockerfile.dockerignore', 'cloudrun-xgboost.requirements.txt'):
        out = destination/'docker'/name
        out.parent.mkdir(exist_ok=True)
        shutil.copyfile(repository/'docker'/name, out)
    shutil.copyfile(repository/'infra/cloudbuild-xgboost.yaml', destination/'cloudbuild.yaml')
    # Do not inherit the repository .gitignore: it deliberately ignores model files.
    (destination/'.gcloudignore').write_text('.gcloudignore\n**/__pycache__/\n**/*.pyc\n')
    receipt = {'artifacts': {name: {'sha256': store.specification['artifacts'][name]['sha256'],
                                 'size_bytes': store.specification['artifacts'][name]['size_bytes']} for name in ARTIFACTS},
               'purpose': 'private XGBoost-only Cloud Build source; not for Git'}
    (destination/'bundle-receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('outputs/forecasting-v1/artifacts'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.artifacts, args.output, Path(__file__).resolve().parents[3]), indent=2))


if __name__ == '__main__': main()
