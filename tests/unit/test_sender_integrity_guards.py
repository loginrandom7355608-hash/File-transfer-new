from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from app.models.manifest import FileManifestEntry, TransferManifest
from app.transfer.receiver_service import ReceiverService
from app.transfer.sender_service import SenderService


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_manifest(relative_path: str, size_bytes: int, mtime_ns: int) -> TransferManifest:
    entry = FileManifestEntry(
        relative_path=relative_path,
        category="videos",
        extension=Path(relative_path).suffix,
        size_bytes=size_bytes,
        modified_time_ns=mtime_ns,
    )
    return TransferManifest(
        session_id="session-guard",
        protocol_version=1,
        source_root="",
        selected_categories=["videos"],
        files=[entry],
    )


def test_sender_refuses_file_that_shrank_since_scan(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "dest"
    source_root.mkdir()
    destination_root.mkdir()

    source_file = source_root / "movie.mp4"
    source_file.write_bytes(b"x" * 1000)

    # Manifest claims a larger size than what's actually on disk right now,
    # simulating the file having been scanned earlier and then shrunk
    # (e.g. re-encoded, replaced, or truncated) before the transfer started.
    manifest = _make_manifest("movie.mp4", size_bytes=5000, mtime_ns=source_file.stat().st_mtime_ns)

    port = get_free_port()
    receiver = ReceiverService(
        bind_host="127.0.0.1",
        port=port,
        destination_root=destination_root,
        chunk_size=4096,
        duplicate_policy="replace",
    )
    receiver.start()
    time.sleep(0.1)

    sender = SenderService(receiver_host="127.0.0.1", receiver_port=port, chunk_size=4096)

    with pytest.raises(RuntimeError, match="source_file_changed_since_scan"):
        sender.send_manifest(manifest, source_root)

    receiver.stop()

    # Nothing should have been committed to the destination.
    assert not (destination_root / "movie.mp4").exists()


def test_sender_refuses_missing_file(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "dest"
    source_root.mkdir()
    destination_root.mkdir()

    # Manifest references a file that was scanned but has since been deleted
    # or moved away before the transfer started.
    manifest = _make_manifest("gone.mp4", size_bytes=1234, mtime_ns=0)

    port = get_free_port()
    receiver = ReceiverService(
        bind_host="127.0.0.1",
        port=port,
        destination_root=destination_root,
        chunk_size=4096,
        duplicate_policy="replace",
    )
    receiver.start()
    time.sleep(0.1)

    sender = SenderService(receiver_host="127.0.0.1", receiver_port=port, chunk_size=4096)

    with pytest.raises(RuntimeError, match="source_file_unavailable"):
        sender.send_manifest(manifest, source_root)

    receiver.stop()


def test_sender_sends_multiple_mixed_extension_files_in_one_batch(tmp_path: Path) -> None:
    """
    Regression test for the original bug report: a folder containing a mix
    of video extensions (e.g. .mp4 and .mkv) sent together in one batch must
    transfer successfully without the receiver's socket timing out or
    closing early.
    """
    source_root = tmp_path / "source" / "Pirates of the Caribbean"
    destination_root = tmp_path / "dest"
    source_root.mkdir(parents=True)
    destination_root.mkdir()

    files_spec = [
        ("1.mkv", b"a" * 50_000),
        ("2.mkv", b"b" * 60_000),
        ("3.mp4", b"c" * 70_000),
        ("4.mp4", b"d" * 80_000),
        ("5.mkv", b"e" * 90_000),
    ]

    entries = []
    for name, data in files_spec:
        path = source_root / name
        path.write_bytes(data)
        entries.append(
            FileManifestEntry(
                relative_path=f"Pirates of the Caribbean/{name}",
                category="videos",
                extension=Path(name).suffix,
                size_bytes=len(data),
                modified_time_ns=path.stat().st_mtime_ns,
            )
        )

    manifest = TransferManifest(
        session_id="session-mixed",
        protocol_version=1,
        source_root=str(tmp_path / "source"),
        selected_categories=["videos"],
        files=entries,
    )

    port = get_free_port()
    receiver = ReceiverService(
        bind_host="127.0.0.1",
        port=port,
        destination_root=destination_root,
        chunk_size=4096,
        duplicate_policy="replace",
    )
    receiver.start()
    time.sleep(0.1)

    sender = SenderService(receiver_host="127.0.0.1", receiver_port=port, chunk_size=4096)
    sender.send_manifest(manifest, tmp_path / "source")

    receiver.stop()

    for name, data in files_spec:
        out_path = destination_root / "Pirates of the Caribbean" / name
        assert out_path.exists(), f"{name} was not received"
        assert out_path.read_bytes() == data

    assert sorted(sender.last_sent_files) == sorted(e.relative_path for e in entries)
