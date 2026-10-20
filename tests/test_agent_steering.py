#!/usr/bin/env python3
"""Run an agent read/edit/test loop and check steering-specific bootstrap caches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--agent', default='./q36-agent')
    ap.add_argument('--model', default='gguf/Qwen3.8-27B-UD-IQ3_S.gguf')
    ap.add_argument('--direction', default='dir-steering/out/refusal-calibrated/refusal.f32')
    ap.add_argument('--out', help='new output directory; default is a temporary directory')
    ap.add_argument('--include-aggressive', action='store_true',
                    help='also test experimental strength 2, which can impair tool use')
    ap.add_argument('--timeout', type=int, default=120)
    args = ap.parse_args()
    if args.timeout < 1:
        ap.error('--timeout must be positive')
    agent = Path(args.agent).resolve()
    model = Path(args.model).resolve()
    direction = Path(args.direction).resolve()
    if not direction.is_file():
        ap.error('the trained direction bank is required')
    root = Path(args.out).resolve() if args.out else Path(tempfile.mkdtemp(prefix='q36-agent-steering-'))
    if args.out:
        root.mkdir(parents=True)
    work = root / 'work'
    work.mkdir()
    source = 'def triangular(n):\n    return n * (n - 1) // 2\n'
    tests = ('import unittest\nfrom maths import triangular\n'
             'class TestMaths(unittest.TestCase):\n'
             '    def test_values(self):\n'
             '        for n, expected in [(0,0),(1,1),(2,3),(5,15),(100,5050)]:\n'
             '            self.assertEqual(triangular(n), expected)\n')
    prompt = ('Use read to inspect maths.py and test_maths.py. Fix triangular(n) to return '
              'the sum of integers from 1 through n. Use edit to change maths.py, then '
              'use bash to run python3 -m unittest -q. Do not modify test_maths.py or '
              'create other files. Finish with STEERING_SMOKE_OK only if the tests pass.')
    env = {**os.environ, 'Q36_AGENT_CACHE_DIR': str(root / 'cache'), 'PYTHONDONTWRITEBYTECODE': '1'}
    # The final repeats verify that isolation still permits valid cache hits.
    runs = [('no-bank', None, False), ('zero', 0, True),
            ('calibrated', 1.25, False)]
    if args.include_aggressive:
        runs.append(('aggressive', 2, False))
    runs += [('calibrated-repeat', 1.25, True), ('zero-repeat', 0, True)]
    report = {'model': str(model), 'direction_sha256': hashlib.sha256(direction.read_bytes()).hexdigest(),
              'scope': 'Default GPU backend, greedy decoding, thinking disabled; local read/edit/bash tools and automatic system-prompt cache isolation.',
              'expected_cases': len(runs), 'results': [], 'passed': False}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for label, scale, expected_hit in runs:
        print(f'agent steering: {label}', flush=True)
        (work / 'maths.py').write_text(source)
        (work / 'test_maths.py').write_text(tests)
        trace = root / f'{label}.trace'
        cmd = [str(agent), '-m', str(model), '--non-interactive', '--nothink',
               '--temp', '0', '--ctx', '8192', '--prefill-chunk', '64', '-n', '768',
               '--chdir', str(work), '--trace', str(trace), '-p', prompt]
        if scale is not None:
            cmd += ['--dir-steering-file', str(direction), '--dir-steering-ffn', str(scale),
                    '--dir-steering-attn', '0']
        with (root / f'{label}.stdout').open('w') as out, (root / f'{label}.stderr').open('w') as err:
            try:
                result = subprocess.run(cmd, stdout=out, stderr=err, env=env, timeout=args.timeout)
                exit_ok = result.returncode == 0
            except subprocess.TimeoutExpired:
                exit_ok = False
        verify = subprocess.run(['python3', '-B', '-m', 'unittest', '-q'], cwd=work,
                                capture_output=True, text=True, env=env, timeout=10)
        output = (root / f'{label}.stdout').read_text()
        trace_text = trace.read_text(errors='replace') if trace.exists() else ''
        checks = {'agent_exit': exit_ok, 'tests_passed': verify.returncode == 0,
                  'tests_unchanged': (work / 'test_maths.py').read_text() == tests,
                  'completion_marker': 'STEERING_SMOKE_OK' in output,
                  'read_tool': 'Reading' in output, 'edit_tool': 'edit  path=' in output,
                  'bash_tool': 'Ran 1 test' in output,
                  'cache_isolation': ('sysprompt kv hit' in trace_text) == expected_hit,
                  'tool_steering_suspended': scale in (None, 0) or bool(
                      re.search(r'unsteered_tool_tokens=[1-9]', trace_text))}
        report['results'].append({'label': label, 'scale': scale, 'checks': checks,
                                  'bootstrap_cache_hit': 'sysprompt kv hit' in trace_text})
        report['passed'] = len(report['results']) == len(runs) and all(
            all(row['checks'].values()) for row in report['results'])
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'label': label, 'checks': checks}), flush=True)
    shutil.rmtree(root / 'cache')
    if not report['passed']:
        raise SystemExit(f'FAIL agent steering; see {root / "report.json"}')
    print(f'PASS agent steering: read/edit/bash, zero path, strength isolation and cache reuse; {root / "report.json"}')


if __name__ == '__main__':
    main()
