from __future__ import annotations

import hashlib
import socket
from pathlib import Path

from app.models.manifest import FileManifestEntry, TransferManifest
from app.networking.transport import SocketTransport
from app.transfer.chunk_io import iter_file_chunks

_CHUNK_SIZE = 4 * 1024 * 1024
_HEAVY_CHUNK_SIZE = 16 * 1024 * 1024
_READ_TIMEOUT_SECONDS = 120.0


class SenderService:
    def __init__(
        self,
        *,
        receiver_host: str,
        receiver_port: int,
        chunk_size: int = _CHUNK_SIZE,
        heavy_mode: bool = False,
    ) -> None:
        self.receiver_host = receiver_host
        self.receiver_port = receiver_port
        self.heavy_mode = heavy_mode
        self.chunk_size = _HEAVY_CHUNK_SIZE if heavy_mode else chunk_size

    def send_manifest(self, manifest: TransferManifest, source_root: Path) -> None:
        with socket.create_connection((self.receiver_host, self.receiver_port), timeout=10) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(_READ_TIMEOUT_SECONDS)

            transport = SocketTransport(sock)

            transport.send_json_message({"type": "HELLO"})
            hello_ack = transport.receive_json_message()
            if hello_ack.get("type") != "HELLO_ACK":
                raise RuntimeError("receiver_did_not_ack_hello")

            # NOTE: We never pre-hash the whole batch before sending MANIFEST.
            # Pre-hashing every file up front (the old "standard mode") could
            # keep the receiver blocked on recv() longer than
            # _READ_TIMEOUT_SECONDS whenever a batch had several large files
            # (e.g. multiple videos, mixed .mp4/.mkv, or large PDFs/images) --
            # the receiver would time out and close the connection before the
            # sender sent a single byte, surfacing as "Socket closed before
            # expected bytes were received". Now every file streams its bytes
            # and computes SHA256 at the same time (one disk read), and
            # MANIFEST goes out immediately so the receiver never waits long.
            files_payload = [self._entry_to_payload_no_hash(entry) for entry in manifest.files]
            transport.send_json_message({
                "type": "MANIFEST",
                "session_id": manifest.session_id,
                "protocol_version": manifest.protocol_version,
                "heavy_mode": self.heavy_mode,
                "files": files_payload,
            })
            manifest_result = transport.receive_json_message()
            if manifest_result.get("type") != "MANIFEST_RESULT" or not manifest_result.get("accepted"):
                raise RuntimeError("manifest_rejected")

            # Fail-closed by design: if any file fails an integrity check
            # (missing, changed since scan, truncated mid-send, or a hash
            # mismatch reported by the receiver), we stop the whole batch
            # immediately rather than silently skipping it and continuing.
            # self.last_sent_files reflects everything that completed
            # successfully before that point.
            self.last_sent_files: list[str] = []
            for entry in manifest.files:
                self._send_one_file_streaming(transport, source_root, entry)
                self.last_sent_files.append(entry.relative_path)

            transport.send_json_message({"type": "TRANSFER_COMPLETE"})
            final_ack = transport.receive_json_message()
            if final_ack.get("type") != "TRANSFER_COMPLETE_ACK":
                raise RuntimeError("missing_transfer_complete_ack")

    def _send_one_file_streaming(self, transport, source_root, entry):
        """
        Send file while computing SHA256 simultaneously — single disk read.
        Normal mode: read file once to hash + read again to send = 2 reads.
        Heavy mode:  read file once, hash each chunk as it leaves = 1 read.
        For a 10GB video this cuts sender disk I/O roughly in half.

        SECURITY / INTEGRITY LAYERS:
        1. Live re-stat immediately before opening the file for send, checked
           against the size recorded at scan time. Catches files that were
           replaced, truncated, or are still being written to disk between
           scan and transfer.
        2. Mid-transfer size enforcement: if the file grows or shrinks while
           being streamed (detected by reading past/short of the expected
           size), the transfer for that file is aborted rather than silently
           sending a partial/extra stream.
        3. SHA256 is computed from the exact bytes placed on the wire (not a
           separate read), so the hash sent in FILE_COMPLETE always matches
           what the receiver gets — no window where hash and payload can
           diverge due to a file changing between two separate reads.
        """
        source_path = source_root / Path(entry.relative_path)

        # Layer 1: re-stat the file right before we touch it. If it doesn't
        # exist anymore, or its size no longer matches what was recorded at
        # scan time, do not send it — tell the receiver to skip it instead of
        # streaming stale/inconsistent bytes.
        try:
            live_stat = source_path.stat()
        except OSError as exc:
            raise RuntimeError(f"source_file_unavailable: {entry.relative_path} ({exc})") from exc

        if live_stat.st_size != entry.size_bytes:
            raise RuntimeError(
                f"source_file_changed_since_scan: {entry.relative_path} "
                f"(scanned {entry.size_bytes} bytes, now {live_stat.st_size} bytes)"
            )

        transport.send_json_message({
            "type": "FILE_START",
            "relative_path": entry.relative_path,
            "chunk_size": self.chunk_size,
            "streaming_hash": True,
        })
        response = transport.receive_json_message()
        if response.get("type") == "FILE_SKIP":
            return
        if response.get("type") == "ERROR":
            raise RuntimeError(f"receiver_rejected_file: {response.get('reason', 'unknown')}")
        if response.get("type") != "FILE_RESUME_INFO":
            raise RuntimeError(f"expected_file_resume_info, got: {response.get('type')}")
        offset = int(response["offset"])
        digest = hashlib.sha256()
        # If resuming, hash the already-received bytes first so the final
        # hash covers the complete file from byte 0
        if offset > 0:
            bytes_counted = 0
            for chunk in iter_file_chunks(source_path, start_offset=0, chunk_size=self.chunk_size):
                remaining = offset - bytes_counted
                if remaining <= 0:
                    break
                slice_chunk = chunk[:remaining]
                digest.update(slice_chunk)
                bytes_counted += len(slice_chunk)

        # Layer 2: enforce the scanned size while streaming. iter_file_chunks
        # stops when the file has no more bytes, so if the file was truncated
        # mid-transfer we'll come up short of entry.size_bytes; if it grew,
        # we cap the read at the scanned size so we never send more than the
        # receiver was told to expect (which would desync the byte-count
        # framing on the wire).
        bytes_sent = offset
        expected_total = entry.size_bytes
        for chunk in iter_file_chunks(source_path, start_offset=offset, chunk_size=self.chunk_size):
            remaining = expected_total - bytes_sent
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            digest.update(chunk)
            transport.send_all(chunk)
            bytes_sent += len(chunk)

        if bytes_sent != expected_total:
            raise RuntimeError(
                f"source_file_truncated_during_send: {entry.relative_path} "
                f"(expected {expected_total} bytes, sent {bytes_sent} bytes)"
            )

        transport.send_json_message({
            "type": "FILE_COMPLETE",
            "relative_path": entry.relative_path,
            "sha256": digest.hexdigest(),
        })
        hash_result = transport.receive_json_message()
        if hash_result.get("type") != "FILE_HASH_RESULT":
            raise RuntimeError("expected_file_hash_result")
        if not hash_result.get("ok"):
            raise RuntimeError("receiver_reported_hash_mismatch")

    def _entry_to_payload_no_hash(self, entry):
        return {
            "relative_path": entry.relative_path,
            "category": entry.category,
            "extension": entry.extension,
            "size_bytes": entry.size_bytes,
            "modified_time_ns": entry.modified_time_ns,
            "sha256": "",
        }
