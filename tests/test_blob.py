from rag.storage.blob import LocalBlobStore, blob_key, sha256_file


def test_same_content_is_stored_once(tmp_path):
    store = LocalBlobStore(tmp_path / "blobs")
    a = tmp_path / "a.PDF"
    b = tmp_path / "copy-of-a.pdf"
    a.write_bytes(b"%PDF-1.7 hello")
    b.write_bytes(b"%PDF-1.7 hello")

    sha_a, key_a = store.put(a)
    sha_b, key_b = store.put(b)

    assert sha_a == sha_b == sha256_file(a)
    assert key_a == key_b == blob_key(sha_a, ".pdf")
    assert key_a.startswith(f"{sha_a[:2]}/{sha_a[2:4]}/")
    assert len(list((tmp_path / "blobs").rglob("*.pdf"))) == 1


def test_fetch_round_trip(tmp_path):
    store = LocalBlobStore(tmp_path / "blobs")
    src = tmp_path / "notes.md"
    src.write_text("# Event loop")
    _, key = store.put(src)

    out = store.fetch(key, tmp_path / "out.md")
    assert out.read_text() == "# Event loop"
    assert store.exists(key)
    assert not store.exists("00/00/missing.md")
