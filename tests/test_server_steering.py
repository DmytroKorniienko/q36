#!/usr/bin/env python3
"""Check server responses, streaming, tools, and steering cache namespaces."""

import argparse
import json
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--server', default='./q36-server')
    ap.add_argument('--model', default='gguf/Qwen3.8-27B-UD-IQ3_S.gguf')
    ap.add_argument('--direction', default='dir-steering/out/refusal-calibrated/refusal.f32')
    ap.add_argument('--out')
    args = ap.parse_args()
    server, model, direction = (Path(p).resolve() for p in (args.server, args.model, args.direction))
    root = Path(args.out).resolve() if args.out else Path(tempfile.mkdtemp(prefix='q36-server-steering-'))
    if args.out:
        root.mkdir(parents=True)
    report = {'results': [], 'passed': False}
    for label, scale in [('normal', None), ('steered', 2), ('normal-again', None)]:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cmd = [str(server), '-m', str(model), '--ctx', '1024', '--prefill-chunk', '64',
               '--host', '127.0.0.1', '--port', str(port), '--kv-disk-dir', str(root / 'cache'),
               '--kv-disk-space-mb', '512', '--kv-cache-min-tokens', '1']
        if scale is not None:
            cmd += ['--dir-steering-file', str(direction), '--dir-steering-ffn', str(scale),
                    '--dir-steering-attn', '0']
        print(f'server steering: {label}', flush=True)
        with (root / f'{label}.log').open('w') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=log)
            try:
                base = f'http://127.0.0.1:{port}'
                deadline = time.monotonic() + 90
                while True:
                    try:
                        with urlopen(base + '/v1/models', timeout=1) as response:
                            json.load(response)
                        break
                    except OSError:
                        assert process.poll() is None, (root / f'{label}.log').read_text()
                        if time.monotonic() > deadline:
                            raise TimeoutError(label)
                        time.sleep(0.2)
                payload = {'model': 'local', 'messages': [{'role': 'user', 'content': 'Reply with only the word READY.'}],
                           'max_tokens': 16, 'temperature': 0, 'reasoning_effort': 'none'}

                def request(body):
                    return urlopen(Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                          headers={'Content-Type': 'application/json'}), timeout=90)

                with request(payload) as response:
                    answer = json.load(response)
                message = answer['choices'][0]['message']['content']
                assert message.strip() == 'READY', answer
                with request({**payload, 'stream': True}) as response:
                    stream = response.read().decode()
                assert 'data: [DONE]' in stream, stream
                chunks = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: {')]
                text = ''.join(c['choices'][0]['delta'].get('content', '') or '' for c in chunks)
                assert text == message, (text, message)
                tools = [{'type': 'function', 'function': {'name': 'echo', 'description': 'Echo a string.',
                         'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}}}]
                body = {**payload, 'messages': [{'role': 'user', 'content': 'Call the echo tool with text STEERING_HTTP_OK.'}],
                        'tools': tools, 'max_tokens': 128}
                with request(body) as response:
                    tool = json.load(response)
                calls = tool['choices'][0]['message'].get('tool_calls', [])
                assert calls and calls[0]['function']['name'] == 'echo', tool
                assert json.loads(calls[0]['function']['arguments']) == {'text': 'STEERING_HTTP_OK'}, tool
                (root / f'{label}.json').write_text(json.dumps({'answer': answer, 'tool': tool, 'stream': stream}, indent=2) + '\n')
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        log_text = (root / f'{label}.log').read_text()
        cache_dir = re.search(r'KV disk cache (.*?) \(budget=', log_text).group(1)
        assert (cache_dir == str(root / 'cache')) == (scale is None), cache_dir
        assert process.returncode == 0, log_text
        report['results'].append({'label': label, 'scale': scale, 'cache_dir': cache_dir,
                                  'nonstream': True, 'stream': True, 'tools': True})
        report['passed'] = len(report['results']) == 3
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.rmtree(root / 'cache')
    print(f'PASS server steering: responses, streaming, tools, cache separation; {root / "report.json"}')


if __name__ == '__main__':
    main()
