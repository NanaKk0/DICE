from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import time

import numpy as np
from ase import Atoms


_MAGIC = b"DGS1"
_ACK_MAGIC = b"ACK1"
_EOF_MAGIC = b"EOF1"
_PREFIX = struct.Struct("!4sQ")
_META_LENGTH = struct.Struct("!I")
_ACK = struct.Struct("!4sQ")
_PROTOCOL = "dice.geometry-stream"
_VERSION = 1


class GeometryStreamError(RuntimeError):
    pass


class GeometryStreamServer:
    """One-frame-at-a-time acknowledged local geometry transport.

    The acknowledgement makes both byte and atom occupancy explicit: at most
    one validated frame is in flight.  A frame is never split.
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        ordered_identities: Sequence[str],
        byte_budget: int = 64 * 1024 * 1024,
        atom_budget: int = 1_000_000,
        timeout_seconds: float = 300.0,
    ) -> None:
        identities = tuple(str(item) for item in ordered_identities)
        if any(not item for item in identities) or len(set(identities)) != len(identities):
            raise ValueError("geometry stream identities must be unique and non-empty")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (byte_budget, atom_budget)
        ):
            raise ValueError("geometry stream byte and atom budgets must be positive integers")
        if timeout_seconds <= 0:
            raise ValueError("geometry stream timeout must be positive")
        requested_socket_path = Path(socket_path).resolve()
        self._short_socket_dir: Path | None = None
        if len(os.fsencode(requested_socket_path)) >= 104:
            self._short_socket_dir = Path(
                tempfile.mkdtemp(prefix="dice-geometry-stream-")
            ).resolve()
            self.socket_path = self._short_socket_dir / "stream.sock"
        else:
            self.socket_path = requested_socket_path
        self.ordered_identities = identities
        self.byte_budget = int(byte_budget)
        self.atom_budget = int(atom_budget)
        self.timeout_seconds = float(timeout_seconds)
        self.max_inflight_bytes = 0
        self.max_inflight_atoms = 0
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.settimeout(self.timeout_seconds)
        try:
            self._server.bind(str(self.socket_path))
            self._server.listen(1)
        except BaseException:
            self._server.close()
            self.socket_path.unlink(missing_ok=True)
            if self._short_socket_dir is not None:
                self._short_socket_dir.rmdir()
            raise

    def produce(
        self,
        chunks: Iterable[Sequence[Atoms]],
        *,
        consumer_alive: Callable[[], bool],
    ) -> None:
        connection: socket.socket | None = None
        try:
            while connection is None:
                try:
                    connection, _ = self._server.accept()
                except socket.timeout as exc:
                    if not consumer_alive():
                        raise GeometryStreamError(
                            "geometry stream worker exited before connecting"
                        ) from exc
                    raise TimeoutError("geometry stream timed out waiting for worker") from exc
            connection.settimeout(self.timeout_seconds)
            ordinal = 0
            for raw_chunk in chunks:
                chunk = tuple(raw_chunk)
                if not chunk:
                    raise ValueError("geometry stream producer emitted an empty chunk")
                for frame in chunk:
                    if ordinal >= len(self.ordered_identities):
                        raise ValueError("geometry stream producer exceeded canonical coverage")
                    identity = str(frame.info.get("structure_id", ""))
                    if identity != self.ordered_identities[ordinal]:
                        raise ValueError("geometry stream producer changed canonical order")
                    payload = encode_geometry_frame(frame, ordinal=ordinal)
                    atoms = len(frame)
                    if len(payload) > self.byte_budget:
                        raise ValueError(
                            "geometry stream frame exceeds the bounded byte occupancy"
                        )
                    if atoms > self.atom_budget:
                        raise ValueError(
                            "geometry stream frame exceeds the bounded atom occupancy"
                        )
                    self.max_inflight_bytes = max(self.max_inflight_bytes, len(payload))
                    self.max_inflight_atoms = max(self.max_inflight_atoms, atoms)
                    self._send_record(connection, _MAGIC, payload, consumer_alive)
                    acknowledgement = _recv_exact(connection, _ACK.size)
                    magic, acknowledged = _ACK.unpack(acknowledgement)
                    if magic != _ACK_MAGIC or acknowledged != ordinal:
                        raise GeometryStreamError(
                            "geometry stream consumer acknowledgement is invalid"
                        )
                    ordinal += 1
            if ordinal != len(self.ordered_identities):
                raise ValueError("geometry stream producer reached early EOF")
            terminal = json.dumps(
                {"protocol": _PROTOCOL, "version": _VERSION, "frames": ordinal},
                separators=(",", ":"),
            ).encode("utf-8")
            self._send_record(connection, _EOF_MAGIC, terminal, consumer_alive)
            magic, acknowledged = _ACK.unpack(_recv_exact(connection, _ACK.size))
            if magic != _ACK_MAGIC or acknowledged != ordinal:
                raise GeometryStreamError("geometry stream terminal acknowledgement is invalid")
        except (BrokenPipeError, ConnectionError, socket.timeout) as exc:
            raise GeometryStreamError("geometry stream worker connection failed") from exc
        finally:
            if connection is not None:
                connection.close()

    def close(self) -> None:
        self._server.close()
        self.socket_path.unlink(missing_ok=True)
        if self._short_socket_dir is not None:
            try:
                self._short_socket_dir.rmdir()
            except FileNotFoundError:
                pass

    @staticmethod
    def _send_record(
        connection: socket.socket,
        magic: bytes,
        payload: bytes,
        consumer_alive: Callable[[], bool],
    ) -> None:
        if not consumer_alive():
            raise GeometryStreamError("geometry stream worker exited before input send")
        connection.sendall(_PREFIX.pack(magic, len(payload)))
        connection.sendall(payload)


def consume_geometry_stream(
    socket_path: str | Path,
    *,
    ordered_identities: Sequence[str],
    byte_budget: int,
    atom_budget: int,
    timeout_seconds: float = 300.0,
) -> Iterator[tuple[Atoms, ...]]:
    identities = tuple(str(item) for item in ordered_identities)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            connection.connect(str(Path(socket_path).resolve()))
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                connection.close()
                raise TimeoutError("geometry stream socket did not appear")
            time.sleep(0.01)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                connection.close()
                raise TimeoutError("geometry stream worker could not connect")
            time.sleep(0.01)
    ordinal = 0
    try:
        while True:
            magic, length = _PREFIX.unpack(_recv_exact(connection, _PREFIX.size))
            if length > byte_budget:
                raise ValueError("geometry stream record length exceeds byte budget")
            payload = _recv_exact(connection, length)
            if magic == _EOF_MAGIC:
                try:
                    terminal = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError("geometry stream terminal record is malformed") from exc
                if terminal != {
                    "protocol": _PROTOCOL,
                    "version": _VERSION,
                    "frames": ordinal,
                } or ordinal != len(identities):
                    raise ValueError("geometry stream terminal coverage is invalid")
                connection.sendall(_ACK.pack(_ACK_MAGIC, ordinal))
                return
            if magic != _MAGIC:
                raise ValueError("geometry stream record magic is invalid")
            frame, actual_ordinal = decode_geometry_frame(
                payload, byte_budget=byte_budget, atom_budget=atom_budget
            )
            if actual_ordinal != ordinal or ordinal >= len(identities):
                raise ValueError("geometry stream ordinal coverage is invalid")
            if str(frame.info.get("structure_id", "")) != identities[ordinal]:
                raise ValueError("geometry stream identity order is invalid")
            yield (frame,)
            connection.sendall(_ACK.pack(_ACK_MAGIC, ordinal))
            ordinal += 1
    except (BrokenPipeError, ConnectionError, socket.timeout) as exc:
        raise GeometryStreamError("geometry stream producer connection failed") from exc
    finally:
        connection.close()


def encode_geometry_frame(frame: Atoms, *, ordinal: int) -> bytes:
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise ValueError("geometry stream ordinal must be a non-negative integer")
    numbers = np.ascontiguousarray(frame.numbers, dtype="<i4")
    coordinates = np.ascontiguousarray(frame.positions, dtype="<f8")
    cell = np.ascontiguousarray(frame.cell.array, dtype="<f8")
    pbc = np.ascontiguousarray(frame.pbc, dtype="u1")
    identity = str(frame.info.get("structure_id", ""))
    if not identity:
        raise ValueError("geometry stream frame requires canonical identity")
    metadata = {
        "protocol": _PROTOCOL,
        "version": _VERSION,
        "ordinal": ordinal,
        "identity": identity,
        "atom_count": len(frame),
        "numbers_dtype": "<i4",
        "numbers_shape": [len(frame)],
        "coordinates_dtype": "<f8",
        "coordinates_shape": [len(frame), 3],
        "cell_dtype": "<f8",
        "cell_shape": [3, 3],
        "pbc_dtype": "|u1",
        "pbc_shape": [3],
    }
    encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return b"".join(
        (_META_LENGTH.pack(len(encoded)), encoded, numbers.tobytes(), coordinates.tobytes(), cell.tobytes(), pbc.tobytes())
    )


def decode_geometry_frame(
    payload: bytes, *, byte_budget: int, atom_budget: int
) -> tuple[Atoms, int]:
    if len(payload) > byte_budget or len(payload) < _META_LENGTH.size:
        raise ValueError("geometry stream payload length is invalid")
    metadata_length = _META_LENGTH.unpack(payload[: _META_LENGTH.size])[0]
    metadata_stop = _META_LENGTH.size + metadata_length
    if metadata_stop > len(payload):
        raise ValueError("geometry stream metadata is truncated")
    try:
        metadata = json.loads(payload[_META_LENGTH.size : metadata_stop])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("geometry stream metadata is malformed") from exc
    if not isinstance(metadata, dict):
        raise ValueError("geometry stream metadata must be an object")
    count = metadata.get("atom_count")
    ordinal = metadata.get("ordinal")
    identity = metadata.get("identity")
    expected_fields = {
        "protocol": _PROTOCOL,
        "version": _VERSION,
        "ordinal": ordinal,
        "identity": identity,
        "atom_count": count,
        "numbers_dtype": "<i4",
        "numbers_shape": [count],
        "coordinates_dtype": "<f8",
        "coordinates_shape": [count, 3],
        "cell_dtype": "<f8",
        "cell_shape": [3, 3],
        "pbc_dtype": "|u1",
        "pbc_shape": [3],
    }
    if metadata != expected_fields:
        raise ValueError("geometry stream dtype, shape, topology, or version is invalid")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or count > atom_budget
        or not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or ordinal < 0
        or not isinstance(identity, str)
        or not identity
    ):
        raise ValueError("geometry stream ordinal, identity, or atom count is invalid")
    number_bytes = count * 4
    coordinate_bytes = count * 3 * 8
    expected_length = metadata_stop + number_bytes + coordinate_bytes + 9 * 8 + 3
    if len(payload) != expected_length:
        raise ValueError("geometry stream numeric payload is truncated or oversized")
    cursor = metadata_stop
    numbers = np.frombuffer(payload, dtype="<i4", count=count, offset=cursor).copy()
    cursor += number_bytes
    coordinates = np.frombuffer(payload, dtype="<f8", count=count * 3, offset=cursor).reshape(count, 3).copy()
    cursor += coordinate_bytes
    cell = np.frombuffer(payload, dtype="<f8", count=9, offset=cursor).reshape(3, 3).copy()
    cursor += 9 * 8
    pbc = np.frombuffer(payload, dtype="u1", count=3, offset=cursor).astype(bool)
    if np.any(numbers <= 0) or not np.isfinite(coordinates).all() or not np.isfinite(cell).all():
        raise ValueError("geometry stream numeric topology data is invalid")
    frame = Atoms(numbers=numbers, positions=coordinates, cell=cell, pbc=pbc)
    frame.info["structure_id"] = identity
    return frame, ordinal


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        part = connection.recv(length - len(chunks))
        if not part:
            raise GeometryStreamError("geometry stream ended mid-record")
        chunks.extend(part)
    return bytes(chunks)


__all__ = [
    "GeometryStreamError",
    "GeometryStreamServer",
    "consume_geometry_stream",
    "decode_geometry_frame",
    "encode_geometry_frame",
]
