"""Download setup uses the upstream manifest without checking a copy into source."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("corrupt", [False, True])
def test_manifest_is_verified_before_downloading_weights(
    tmp_path, monkeypatch, corrupt
):
    from models.cua_s1.multimodal import model

    raw = json.dumps(
        {
            "artifacts": [
                {
                    "repo_id": "test/base",
                    "revision": "fixed",
                    "name": "Qwen3.5-4B",
                    "files": {"config.json": {}},
                }
            ]
        }
    ).encode()
    import hashlib

    monkeypatch.setattr(
        model, "WEIGHTS_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest()
    )
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        kwargs["local_dir"].mkdir(parents=True)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )
    path = Path(__file__).resolve().parents[2] / "recipe/cua_s1/download_weights.py"
    spec = importlib.util.spec_from_file_location("download_weights_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = raw + b" " if corrupt else raw
    urls = []

    def fetch(url, timeout):
        urls.append(url)
        assert timeout == 30
        return io.BytesIO(payload)

    monkeypatch.setattr(module, "urlopen", fetch)
    verified = []

    def verify(base, adapter):
        assert (base.parent / "weights.lock.json").read_bytes() == raw
        verified.append((base, adapter))

    monkeypatch.setattr(module, "verify_weights", verify)
    dest = tmp_path / "weights"
    monkeypatch.setattr(sys, "argv", [str(path), "--dest", str(dest)])
    if corrupt:
        with pytest.raises(ValueError, match="manifest checksum"):
            module.main()
        assert not downloads and not verified and not dest.exists()
    else:
        module.main()
        assert downloads == [
            {
                "repo_id": "test/base",
                "revision": "fixed",
                "local_dir": dest / "Qwen3.5-4B",
                "allow_patterns": ["config.json"],
                "token": False,
            }
        ]
        assert verified == [(dest / "Qwen3.5-4B", dest / "cua-s1-4b-0.2/multimodal")]
    assert model.REFERENCE_REVISION in urls[0]
