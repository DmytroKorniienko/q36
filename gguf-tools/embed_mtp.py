#!/usr/bin/env python3
"""Append a Qwen3.6 MoE MTP block without requantizing the target weights."""

import argparse
import os
import struct


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


class GGUF:
    def __init__(self, path, size=None):
        self.path = str(path)
        self.size = size if size is not None else os.path.getsize(path)
        with open(path, 'rb') as f:
            def read(fmt):
                return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))

            def text():
                return f.read(read('Q')[0]).decode()

            def value(kind):
                if kind == 8:
                    return text()
                if kind == 9:
                    item, count = read('IQ')
                    if item == 8:
                        for _ in range(count):
                            f.seek(read('Q')[0], 1)
                    else:
                        f.seek(count * struct.calcsize(formats[item]), 1)
                    return None
                return read(formats[kind])[0]

            formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                       6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
            magic, version, nt, nk = read('4sIQQ')
            if magic != b'GGUF' or version != 3:
                raise ValueError('expected GGUF v3')
            self.kv = {}
            for _ in range(nk):
                begin = f.tell()
                name = text()
                kind = read('I')[0]
                val = value(kind)
                end = f.tell()
                f.seek(begin)
                self.kv[name] = (kind, val, f.read(end - begin))
            self.tensors = []
            for _ in range(nt):
                name = text()
                rank = read('I')[0]
                dims = read('Q' * rank)
                kind, offset = read('IQ')
                self.tensors.append(dict(name=name, dims=dims, kind=kind, offset=offset))
            self.alignment = self.kv.get('general.alignment', (4, 32))[1]
            self.start = align(f.tell(), self.alignment)
        ordered = sorted(self.tensors, key=lambda t: t['offset'])
        for i, tensor in enumerate(ordered):
            end = ordered[i + 1]['offset'] if i + 1 < len(ordered) else self.size - self.start
            tensor['span'] = end - tensor['offset']
            tensor['path'] = self.path
            tensor['absolute'] = self.start + tensor['offset']


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def u32(key, value):
    return 4, value, string(key) + struct.pack('<II', 4, value)


def write_gguf(path, metadata, tensors, alignment=32):
    info = []
    offset = 0
    for tensor in tensors:
        dims = tensor['dims']
        info.append(string(tensor['name']) + struct.pack('<I', len(dims)) +
                    struct.pack('<' + 'Q' * len(dims), *dims) +
                    struct.pack('<IQ', tensor['kind'], offset))
        offset += align(tensor['span'], alignment)
    header = (struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), len(metadata)) +
              b''.join(v[2] for v in metadata.values()) + b''.join(info))
    with open(path, 'xb') as out:
        out.write(header)
        out.write(bytes(align(len(header), alignment) - len(header)))
        for tensor in tensors:
            with open(tensor['path'], 'rb') as source:
                source.seek(tensor['absolute'])
                remaining = tensor['span']
                while remaining:
                    data = source.read(min(remaining, 8 << 20))
                    if not data:
                        raise ValueError('truncated tensor: ' + tensor['name'])
                    out.write(data)
                    remaining -= len(data)
            out.write(bytes(align(tensor['span'], alignment) - tensor['span']))


def embed(base_path, head_path, out_path):
    base, head = GGUF(base_path), GGUF(head_path)
    if base.kv.get('general.architecture', (None, None))[1] != 'qwen35moe':
        raise ValueError('expected qwen35moe target')
    trunk = [t for t in base.tensors if not t['name'].startswith('blk.40.')]
    mtp = [t for t in head.tensors if t['name'].startswith('blk.40.')]
    if len(mtp) != 20 or len({t['name'] for t in mtp}) != 20:
        raise ValueError('expected 20 separate blk.40 MTP tensors')
    if sum(t['span'] for t in mtp) >= 500_000_000:
        raise ValueError('MTP block must be under 500 MB')
    metadata = {k: v for k, v in base.kv.items() if not k.startswith('split.')}
    metadata['qwen35moe.block_count'] = u32('qwen35moe.block_count', 41)
    metadata['qwen35moe.nextn_predict_layers'] = u32('qwen35moe.nextn_predict_layers', 1)
    write_gguf(out_path, metadata, trunk + mtp, base.alignment)
    print(f'MTP: {sum(t["span"] for t in mtp):,} bytes; target tensors copied unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    embed(args.base, args.head, args.out)
