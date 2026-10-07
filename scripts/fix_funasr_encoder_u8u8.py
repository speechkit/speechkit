#!/usr/bin/env python3
"""Convert the tested FunASR Nano INT8 encoder to U8U8 using only the standard library.

See docs/funasr-nano.md for the affected CPU kernels, installation, and rollback.
The input is checked against the tested export before a new output file is made.
"""

import argparse
import hashlib
import sys
from pathlib import Path


# sherpa-onnx-funasr-nano-int8-2025-12-30, encoder_adaptor.int8.onnx.
SOURCE_SHA256 = "d0246c823f2c34133ae0efee395d8a189c8f92643e3432f866939ee34d34492c"
CONVERTED_SHA256 = "29e0b6796fc7fed4440561aa31a70f7087915bf23dafbea59a382ca9bb3bd623"
SHIFT_BYTES = bytes(i ^ 128 for i in range(256))


def _varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ValueError("protobuf varint exceeds 64 bits")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("invalid protobuf varint")


def _fields(data):
    """Read ONNX's protobuf fields while preserving each original encoding."""
    # Schema: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
    offset = 0
    while offset < len(data):
        start = offset
        key, offset = _varint(data, offset)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("invalid protobuf field number")
        if wire == 0:
            payload, offset = _varint(data, offset)
        else:
            if wire == 2:
                length, offset = _varint(data, offset)
            elif wire in (1, 5):
                length = 8 if wire == 1 else 4
            else:
                raise ValueError(f"unsupported protobuf wire type {wire}")
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf field")
            payload = data[offset:end]
            offset = end
        yield number, wire, payload, data[start:offset]


def _value(data, number):
    return next((p for n, _, p, _ in _fields(data) if n == number), None)


def _integer(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number, payload):
    if isinstance(payload, int):
        return _integer(number << 3) + _integer(payload)
    return _integer((number << 3) | 2) + _integer(len(payload)) + bytes(payload)


def _shift_integer(value):
    # Negative int32_data values use sign-extended, 64-bit protobuf varints.
    if value >= 1 << 63:
        value -= 1 << 64
    if not -128 <= value <= 127:
        raise ValueError(f"invalid INT8 value: {value}")
    return _integer(value + 128)


def _shift_tensor(tensor):
    if _value(tensor, 2) != 3:  # TensorProto.INT8
        raise ValueError("expected an INT8 weight or zero point")
    parts = []
    for number, wire, payload, raw in _fields(tensor):
        if number == 2:
            parts.append(_field(2, 2))  # TensorProto.UINT8
        elif number == 9:  # raw_data: signed bytes -> unsigned bytes, +128.
            parts.append(_field(9, bytes(payload).translate(SHIFT_BYTES)))
        elif number == 5:  # int32_data: zero points can be packed or unpacked.
            if wire == 0:
                parts.append(_integer(5 << 3) + _shift_integer(payload))
            elif wire == 2:
                offset = 0
                packed = bytearray()
                while offset < len(payload):
                    item, offset = _varint(payload, offset)
                    packed.extend(_shift_integer(item))
                parts.append(_field(5, packed))
            else:
                raise ValueError("invalid int32_data encoding")
        else:
            parts.append(raw)
    return _field(5, b"".join(parts))  # GraphProto.initializer


def _rewrite_model(data):
    """Change only MatMulInteger's signed weights and their explicit zero points."""
    model_fields = list(_fields(memoryview(data)))
    graph = _value(memoryview(data), 7)
    if graph is None:
        raise ValueError("the ONNX model has no graph")
    graph_fields = list(_fields(graph))
    tensors = {
        bytes(_value(p, 8)).decode(): p
        for n, _, p, _ in graph_fields
        if n == 5
    }
    changed = set()
    count = 0
    for number, _, node, _ in graph_fields:
        if number != 1 or bytes(_value(node, 4)) != b"MatMulInteger":
            continue
        inputs = [bytes(p).decode() for n, _, p, _ in _fields(node) if n == 1]
        if len(inputs) != 4 or not inputs[3]:
            raise ValueError("expected an explicit weight zero point")
        weight = tensors.get(inputs[1])
        zero = tensors.get(inputs[3])
        if weight is None or zero is None:
            raise ValueError("expected constant weights and weight zero points")
        if _value(weight, 2) == 3:
            if _value(zero, 2) != 3:
                raise ValueError("weight and zero point types disagree")
            changed.update((inputs[1], inputs[3]))
            count += 1

    if count == 0:
        raise ValueError("no signed MatMulInteger weights to convert")
    for number, _, node, _ in graph_fields:
        if number == 1:
            for n, _, p, _ in _fields(node):
                if n == 1 and bytes(p).decode() in changed:
                    if bytes(_value(node, 4)) != b"MatMulInteger":
                        raise ValueError("a converted tensor has another consumer")
        if number in (11, 12, 13) and bytes(_value(node, 1)).decode() in changed:
            raise ValueError("a converted tensor also has a graph type declaration")

    graph_parts = []
    for number, _, payload, raw in graph_fields:
        if number == 5 and bytes(_value(payload, 8)).decode() in changed:
            graph_parts.append(_shift_tensor(payload))
        else:
            graph_parts.append(raw)
    parts = []
    for number, _, _, raw in model_fields:
        if number == 7:
            parts.append(_integer((7 << 3) | 2) + _integer(sum(map(len, graph_parts))))
            parts.extend(graph_parts)
        else:
            parts.append(raw)
    return parts, count, len(changed)


def _write_new_file(destination, parts, expected_sha256):
    created = False
    try:
        # Exclusive creation also refuses existing files and dangling symlinks.
        with destination.open("xb") as output:
            created = True
            digest = hashlib.sha256()
            for part in parts:
                output.write(part)
                digest.update(part)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("converted model checksum does not match the tested output")
    except BaseException:
        if created:
            destination.unlink()
        raise


def convert(source, destination):
    """Write the tested encoder conversion to a new path, preserving the source."""
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists; choose a new output path")
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest == CONVERTED_SHA256:
        raise ValueError("this encoder is already U8U8; no conversion is needed")
    if digest != SOURCE_SHA256:
        raise ValueError("unsupported encoder checksum; see docs/funasr-nano.md")
    parts, count, tensors = _rewrite_model(data)
    _write_new_file(destination, parts, CONVERTED_SHA256)
    return count, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original encoder_adaptor.int8.onnx")
    parser.add_argument("destination", type=Path, help="new U8U8 ONNX file; must not exist")
    args = parser.parse_args()
    try:
        count, tensors = convert(args.source, args.destination)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Converted {count} MatMulInteger operations / {tensors} initializers to U8U8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
