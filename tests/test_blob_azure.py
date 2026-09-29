"""Azure Blob backend against Azurite (docker compose --profile azurite up -d azurite)."""
import uuid

import pytest

from rag.storage.blob import AzureBlobStore, blob_key, sha256_file

pytestmark = pytest.mark.integration
# Azurite's documented development account (public, not a secret).
AZURITE = ("DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
           "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
           "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;")


@pytest.fixture
def store():
    s = AzureBlobStore(AZURITE, f"test-{uuid.uuid4().hex[:8]}")
    try:
        s.check()                      # creates the container
    except Exception:
        pytest.skip("Azurite not running (docker compose --profile azurite up -d azurite)")
    yield s
    s.container.delete_container()


def test_put_dedup_fetch(store, tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "copy.pdf"
    a.write_bytes(b"%PDF-1.7 same bytes")
    b.write_bytes(b"%PDF-1.7 same bytes")
    sha, key = store.put(a)
    assert store.put(b) == (sha, key) == (sha256_file(a), blob_key(sha, ".pdf"))
    assert len(list(store.container.list_blobs())) == 1          # stored once
    out = store.fetch(key, tmp_path / "out.pdf")
    assert out.read_bytes() == b"%PDF-1.7 same bytes"
    assert store.exists(key) and not store.exists("00/00/missing.pdf")
