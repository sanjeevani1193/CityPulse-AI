"""Content-checked intermediate audits; these never constitute final publication."""

from pathlib import Path
import uuid

from citypulse_ai.etl.archives import check_disk, fingerprint, write_json
from citypulse_ai.etl.storage_state import revision_for


class Checkpoints:
    def __init__(self, root: Path, identity: str, budget: int, reserve: int):
        self.root = root
        self.identity, self.budget, self.reserve = identity, budget, reserve
        if root.is_symlink():
            raise ValueError("Checkpoint directory cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.guard()
        used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        check_disk(root, max(0, budget-used), reserve)

    def guard(self):
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if used > self.budget:
            raise ValueError("Reconciliation checkpoint disk budget exceeded")
        check_disk(self.root, 0, self.reserve)

    def asset(self, label: str) -> Path:
        # Interrupted files are preserved; a new attempt never overwrites them.
        return self.root / f"{label}-{uuid.uuid4().hex}.parquet"

    def load(self, name: str):
        import json
        path = self.root / f"{name}.json"
        if not path.exists():
            return None
        bundle = json.loads(path.read_text())
        payload = bundle['payload']
        if (path.is_symlink() or bundle['sha256'] != revision_for(payload)
                or payload.get('format_version') != 1
                or payload['identity'] != self.identity or payload['stage'] != name):
            raise ValueError(f"Changed reconciliation checkpoint: {path}")
        for receipt in payload['files']:
            asset = self.root / receipt['name']
            if (Path(receipt['name']).name != receipt['name'] or asset.is_symlink()
                    or not asset.is_file() or asset.stat().st_size != receipt['size_bytes']
                    or fingerprint(asset) != receipt['sha256']):
                raise ValueError(f"Changed reconciliation checkpoint asset: {asset}")
        self.guard()
        return payload

    def save(self, name: str, data: dict, files=()):
        self.guard()
        payload = {'format_version': 1, 'identity': self.identity, 'stage': name, 'data': data,
                   'files': [{'name': p.name, 'size_bytes': p.stat().st_size,
                              'sha256': fingerprint(p)} for p in files]}
        if (self.root / f"{name}.json").exists():
            raise ValueError(f"Checkpoint already exists: {name}")
        write_json(self.root / f"{name}.json", {'sha256': revision_for(payload), 'payload': payload})
        self.guard()
        return payload
