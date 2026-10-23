import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location('embed_mtp', Path(__file__).resolve().parents[1] / 'embed_mtp.py')
mtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtp)


def payload(g, tensor):
    with open(g.path, 'rb') as f:
        f.seek(tensor['absolute'])
        return f.read(tensor['span'])


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / 'raw.bin'
        raw.write_bytes(struct.pack('<512H', *([0x3e80, 0xbf00] * 256)))
        metadata = {'general.architecture': (8, 'qwen35moe',
                    mtp.string('general.architecture') + struct.pack('<I', 8) + mtp.string('qwen35moe')),
                    'qwen35moe.block_count': mtp.u32('qwen35moe.block_count', 40)}
        names = ['attn_k.weight', 'attn_k_norm.weight', 'attn_norm.weight',
                 'attn_output.weight', 'attn_q.weight', 'attn_q_norm.weight', 'attn_v.weight',
                 'ffn_down_exps.weight', 'ffn_down_shexp.weight', 'ffn_gate_exps.weight',
                 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight', 'ffn_gate_shexp.weight',
                 'ffn_up_exps.weight', 'ffn_up_shexp.weight', 'nextn.eh_proj.weight',
                 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight',
                 'post_attention_norm.weight']
        tensors = [dict(name='blk.40.' + name, dims=(8,), kind=0, span=32,
                        path=str(raw), absolute=0) for name in names]
        for tensor in tensors:
            if '_exps.' in tensor['name']:
                tensor.update(dims=(256, 1, 2), kind=30, span=1024)
        source = root / 'head-bf16.gguf'
        mtp.write_gguf(source, metadata, tensors)
        for gate_type, gate_id, down_type, down_id in [('q2_k', 10, 'q4_k', 12),
                                                      ('q4_k', 12, 'q2_k', 10)]:
            head = root / (gate_type + '.gguf')
            subprocess.run([binary, '--in', str(source), '--out', str(head),
                            '--mtp-gate-up', gate_type, '--mtp-down', down_type,
                            '--allow-synthetic-imatrix'], check=True, capture_output=True)
            h = mtp.GGUF(head)
            for tensor in h.tensors:
                if 'ffn_down_exps' in tensor['name']:
                    assert tensor['kind'] == down_id
                elif 'ffn_gate_exps' in tensor['name'] or 'ffn_up_exps' in tensor['name']:
                    assert tensor['kind'] == gate_id
                else:
                    assert tensor['kind'] == 0
                    assert payload(h, tensor) == raw.read_bytes()[:32]
            base = root / ('base-' + gate_type + '.gguf')
            trunk = dict(name='output_norm.weight', dims=(8,), kind=0,
                         span=32, path=str(raw), absolute=32)
            mtp.write_gguf(base, metadata, [trunk])
            output = root / ('embedded-' + gate_type + '.gguf')
            mtp.embed(base, head, output)
            b, result = mtp.GGUF(base), mtp.GGUF(output)
            assert result.kv['qwen35moe.block_count'][1] == 41
            assert result.kv['qwen35moe.nextn_predict_layers'][1] == 1
            assert payload(result, result.tensors[0]) == payload(b, b.tensors[0])
            for src, dst in zip(h.tensors, result.tensors[1:]):
                assert (src['name'], src['kind'], src['dims']) == (dst['name'], dst['kind'], dst['dims'])
                assert payload(h, src) == payload(result, dst)
            try:
                mtp.embed(base, head, output)
            except FileExistsError:
                pass
            else:
                raise AssertionError('overwrote an existing model')
            try:
                mtp.embed(base, base, root / 'invalid.gguf')
            except ValueError:
                pass
            else:
                raise AssertionError('accepted a missing MTP block')
    print('MTP recipes, protected tensors, grafted payloads and output protection: OK')


if __name__ == '__main__':
    main()
