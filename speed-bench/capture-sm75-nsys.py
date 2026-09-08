#!/usr/bin/env python3
"""Opt-in Nsight capture of one frozen GPU1 reproducer; no build or retry.

--preflight-only executes host collectors and profiler metadata queries, never
the GPU executable. Actual execution requires --run-frozen-gpu1 and the runner's
exact environment command. Profiler success is not GPU stability clearance.
"""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time
import importlib.util


def sibling(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = sibling('capture-sm75-gpu1-failure')
retention = sibling('qualify-sm75-nsys-retention')
host, gate = retention.host, retention.gate
RECEIPT_SHA = '47cf0ba76e0d2a3f535af483106ea23f3ff38f5c6daee59a179299c833cd14c1'
RUNTIME = {
    '/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07': '4839b5da17cd8f58a8e9c57e97c9b61f3c41be7934479f48afc983416b2492c3',
    '/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.595.91.07': '003d6f118e9ff0b886a03ef2e8b1a6d511e8aa9e42be292db469bbc0ac84b5d5',
    '/usr/lib/x86_64-linux-gnu/libnvidia-nvvm70.so.4': '64e62bd2f763c575418bb7d660e8715c57d49ef66640a2385896324238896acd',
    '/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.595.91.07': 'e933f11b1d5f99cc43c6764d79652fddc06f40a8245538b2190521a3c5da6dd1',
    '/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublas.so.13.4.1.3': 'd089edf0a70ba75f0f422927da121f59018247f491741a3bf0b20637ed7c7b61',
    '/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublasLt.so.13.4.1.3': '8d702c94d90bd4bd0c032fd4207ae655b4a2d036bd31920c1fbf4ecaf3547c59',
    '/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcudart.so.13.2.86': '862020da24ec4db58470c8dfad08e8753b417c459c115326edf15e9f574c92e3',
}
CUDA_OPTIONS = ('--trace=cuda-sw,cublas-verbose', '--cuda-trace-scope=process-tree',
                '--cuda-trace-all-apis=true', '--cuda-memory-usage=true',
                '--cuda-event-trace=false', '--cuda-flush-interval=100')
CORE_LIBS = ('libcuda.so.', 'libcudart.so.', 'libcublas.so.', 'libcublasLt.so.')


def exact_environment(command, executable, inherited):
    argv, environment = base.trace_child_command(command, executable, inherited)
    # Wrapper selectors are not profiler controls and must not reach the target.
    for key in ('NSYS_CAPTURE', 'NSYS_PREFLIGHT_ONLY', 'NSYS_QUALIFICATION_ARCHIVE'):
        environment.pop(key, None)
    for key, value in environment.items():
        if value and (key in host.FORBIDDEN_ENV or key.startswith(
                ('NSYS_', 'CUPTI_', 'CUBLAS_LOG', 'NV_COMPUTE_SANITIZER', 'DS4_RUNTIME_TRACE'))):
            raise ValueError('remove unrelated instrumentation: ' + key)
    expected = {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': base.GPU1_UUID,
                'LD_LIBRARY_PATH': '/usr/local/cuda/lib64:'}
    for key in base.RUNTIME_CONTROLS:
        if key in expected:
            if environment.get(key) != expected[key]:
                raise ValueError('runtime control differs from reviewed failure: ' + key)
        elif key in environment:
            raise ValueError('runtime control must remain unset: ' + key)
    return argv, environment


def receipt(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024 or base.sha256(path) != RECEIPT_SHA:
        raise ValueError('qualification archive is not the reviewed 18q7z_z2 receipt')
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if len(members) != 101 or any(not m.isfile() for m in members):
            raise ValueError('unexpected qualification archive structure')
        name = 'sm75-nsys-retention-18q7z_z2/summary.json'
        summary = json.load(archive.extractfile(name))
    if summary['retention_qualification'] != 'passed' or summary['issues']:
        raise ValueError('qualification did not pass')
    for source in (Path(retention.__file__), Path(host.__file__), Path(gate.__file__)):
        if base.sha256(source) != summary['source_sha256'][source.name]:
            raise ValueError('qualified source changed: ' + source.name)
    return summary


def mapped_files(pid):
    lines = (Path('/proc') / str(pid) / 'maps').read_text().splitlines()
    return sorted({line.split(None, 5)[5] for line in lines
                   if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')})


def target_identity(pid, admitted, executable):
    info = host.process_info(pid)
    if info['starttime'] != admitted['starttime'] or info['uid'] != os.getuid():
        raise ValueError('admitted target generation changed')
    directory = Path('/proc') / str(pid)
    if (directory / 'exe').resolve() != executable:
        return False
    argv = [os.fsdecode(x) for x in (directory / 'cmdline').read_bytes().split(b'\0') if x]
    if argv != [str(executable)]:
        raise ValueError('frozen target command line changed')
    return True


class NsightCapture(base.Capture):
    def __init__(self, output, executable, command, case_timeout, qualification, nsys,
                 preflight_only=False):
        super().__init__(output, executable, command, case_timeout, preflight_only)
        self.qualification = Path(qualification)
        self.nsys = Path(nsys).resolve()
        self.profiler = None
        self.prefixes = None
        self.tree = None
        self.admitted = None
        self.target_verified = False
        self.recorded_paths = set()
        self.pinned = dict(RUNTIME)
        self.directory = self.output / 'nsys'
        self.summary['workload_returncode_source'] = 'unavailable-grandchild-not-waited'

    def prepare_runtime_trace(self):
        # Invoked by inherited preflight, before sudo/journal/GPU inventories.
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise ValueError('Linux Python pidfds required')
        qualification = receipt(self.qualification)
        self.argv, self.environment = exact_environment(self.command, self.executable, os.environ)
        if str(self.nsys) != qualification['nsys']['path']:
            raise ValueError('Nsight entry point differs from qualified path')
        self.pinned[str(self.nsys)] = qualification['nsys']['sha256']
        for item in qualification['observed_components']:
            if item.get('sha256') and item['path'].startswith('/opt/nvidia/nsight-systems/'):
                self.pinned[item['path']] = item['sha256']
        self.verify_pins()
        self.directory.mkdir(mode=0o700)
        self.summary.update({'execution_mode': 'instrumented-nsys-frozen-gpu1',
            'qualification_archive_sha256': RECEIPT_SHA, 'pinned_files': self.pinned,
            'effective_runtime_controls': {k: {'set': k in self.environment, 'value': self.environment.get(k)}
                                           for k in base.RUNTIME_CONTROLS},
            'gpu_capture_qualified': False, 'workload_returncode': None})
        for label, options in (('version', ['--version']), ('profile-help', ['profile', '--help']),
                               ('export-help', ['export', '--help'])):
            directory = self.directory / label
            directory.mkdir()
            result = host.run_owned([str(self.nsys), *options], directory, seconds=20)
            if (result['returncode'] != 0 or result['reason'] != 'processes-exited' or
                    result['surviving_owned_pids'] or result['cleanup_issues'] or
                    not result['reader_finished'] or result['reader_errors'] or result['log_overflow']):
                raise RuntimeError('Nsight metadata query failed: ' + label)
        help_text = (self.directory / 'profile-help/console.log').read_text()
        host.validate_help(help_text)
        if 'cuda-sw' not in help_text or 'cublas-verbose' not in help_text:
            raise ValueError('qualified CUDA/cuBLAS trace modes absent')
        # Inventory CUDA-specific bundle files before launch. These are newly
        # observed, not falsely labeled covered by the CPU qualification.
        bundle = Path('/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64')
        inventory = {}
        for path in bundle.rglob('*'):
            if path.is_file() and ('.so' in path.name or path.name in ('nsys', 'nsys-launcher', 'CudaGpuInfoDumper')):
                resolved = path.resolve()
                if not resolved.is_relative_to(bundle) or len(inventory) >= 2048:
                    raise ValueError('profiler bundle inventory path/count limit')
                inventory[str(resolved)] = base.sha256(resolved)
        self.bundle_inventory = inventory
        self.summary['profiler_bundle_prelaunch_inventory'] = inventory
        self.summary['profiler_bundle_coverage'] = 'static files observed before run; not GPU compatibility qualification'
        shutil.copyfile(gate.__file__, self.directory / 'sm75-nsys-launch-gate.py')
        self.gate_hash = base.sha256(self.directory / 'sm75-nsys-launch-gate.py')
        self.save()

    def verify_pins(self):
        for path, expected in self.pinned.items():
            if base.sha256(path) != expected:
                raise ValueError('reviewed executable/library changed: ' + path)
        if base.sha256(self.executable) != base.EXPECTED_SHA256:
            raise ValueError('frozen executable changed')

    def preflight(self):
        super().preflight()
        # Disk hashes alone do not establish the running kernel driver or the
        # /usr/local/cuda alias selected by the historical LD_LIBRARY_PATH.
        version = Path('/proc/driver/nvidia/version').read_text()
        if not re.search(r'\b595\.91\.07\b', version):
            raise ValueError('running NVIDIA module differs from reviewed 595.91.07 stack')
        self.summary['running_nvidia_module'] = version
        library_dir = Path('/usr/local/cuda-13.2/targets/x86_64-linux/lib')
        if Path('/usr/local/cuda/lib64').resolve() != library_dir:
            raise ValueError('/usr/local/cuda/lib64 alias changed')
        aliases = {
            '/usr/lib/x86_64-linux-gnu/libcuda.so.1': '/usr/lib/x86_64-linux-gnu/libcuda.so.595.91.07',
            '/usr/local/cuda/lib64/libcudart.so.13': str(library_dir / 'libcudart.so.13.2.86'),
            '/usr/local/cuda/lib64/libcublas.so.13': str(library_dir / 'libcublas.so.13.4.1.3'),
            '/usr/local/cuda/lib64/libcublasLt.so.13': str(library_dir / 'libcublasLt.so.13.4.1.3'),
        }
        for alias, expected in aliases.items():
            if Path(alias).resolve() != Path(expected):
                raise ValueError('runtime library alias changed: ' + alias)
        self.summary['runtime_aliases'] = aliases
        self.save()

    def stop_reason(self, deadline):
        if self.interrupted.is_set(): return 'interrupted'
        if self.fault.is_set(): return 'stopped-on-kernel-fault'
        if (self.stream_error.is_set() or self.journal_process.poll() is not None or
                not self.journal_reader.is_alive()): return 'stopped-on-collector-failure'
        if time.monotonic() >= deadline: return 'timeout'
        return None

    def observe_mappings(self, pid):
        phase = 'actual_target_mapped_hashes' if self.target_verified else 'gate_mapped_hashes'
        for path in mapped_files(pid):
            key = (phase, path)
            if key in self.recorded_paths:
                continue
            relevant = re.search(r'lib(?:cuda|cublas|cudart|nvidia)|nsight|nsys|ToolsInjection|cupti', path, re.I)
            if not relevant:
                continue
            if path.endswith(' (deleted)'):
                self.issue('mapped component vanished before hashing: ' + path)
                self.recorded_paths.add(key)
                continue
            actual = base.sha256(path)
            self.summary.setdefault(phase, {})[path] = actual
            self.recorded_paths.add(key)
            expected = self.pinned.get(path, self.bundle_inventory.get(path))
            if expected is not None and actual != expected:
                raise ValueError('loaded component hash mismatch: ' + path)
            if re.search(r'/lib(?:cuda|cudart|cublas|nvidia)', path) and path not in RUNTIME:
                raise ValueError('unreviewed CUDA/driver library mapped: ' + path)

    def stop_owned(self):
        if not self.tree:
            if self.profiler and self.profiler.poll() is None:
                self.profiler.kill()  # Only unreaped direct Popen child.
            return
        if self.admitted:
            pid = self.admitted['pid']
            for sig, duration in ((signal.SIGTERM, 1), (signal.SIGKILL, 1)):
                self.tree.send(pid, sig)
                until = time.monotonic() + duration
                while self.tree.alive(pid) and time.monotonic() < until:
                    time.sleep(0.02)
        # Let profiler finalize after target death, but never wait indefinitely.
        until = time.monotonic() + 5
        next_copy = 0
        while self.profiler.poll() is None and time.monotonic() < until:
            if self.prefixes and time.monotonic() >= next_copy:
                try: self.prefixes.capture()
                except Exception as error:
                    self.issue('shutdown prefix retention failed: ' + str(error))
                    break
                next_copy = time.monotonic() + 0.1
            time.sleep(0.05)
        survivors = self.tree.stop()
        self.summary['surviving_owned_pids'] = survivors
        for issue in self.tree.cleanup_issues: self.issue(issue)
        if survivors: self.issue('owned profiler/target processes survived bounded stop')

    def execute(self):
        reason = self.stop_reason(float('inf'))
        if reason: raise RuntimeError(reason + ' before profiler launch')
        self.verify_pins()
        nonce = secrets.token_hex(16)
        config = self.directory / 'gate-config.json'
        gate.private_json(config, {'nonce': nonce, 'executable': str(self.executable),
                                  'sha256': base.EXPECTED_SHA256, 'arguments': []})
        tmp = self.directory / 'live-tmp'
        tmp.mkdir(mode=0o700)
        prefixes = retention.PrefixStore(tmp, self.directory / 'retained-prefixes')
        self.prefixes = prefixes
        gate_path = self.directory / 'sm75-nsys-launch-gate.py'
        gate_argv = [sys.executable, '-I', str(gate_path), str(config)]
        options = [o for o in host.HOST_OPTIONS if not o.startswith('--trace=')]
        argv = [str(self.nsys), 'profile', *options, *CUDA_OPTIONS,
                '--session-new=sm75_gpu1_' + nonce, '--output=' + str(self.directory / 'report'), *gate_argv]
        environment = dict(self.environment, NSYS_TMPDIR=str(tmp), TMPDIR=str(tmp))
        self.summary['profiler_command'] = argv
        self.summary['workload'] = 'waiting-for-admission'
        overflow = threading.Event()
        self.profiler = subprocess.Popen(argv, env=environment, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        self.summary['profiler_pid'] = self.profiler.pid
        reader = None
        try:
            self.tree = host.OwnedTree(self.profiler.pid)
            def drain():
                size = 0
                try:
                    with self.profiler.stdout, (self.directory / 'console.log').open('xb') as log, \
                            (self.output / 'application-timeline.jsonl').open('x') as timeline:
                        while data := self.profiler.stdout.read1(16384):
                            log.write(data[:max(0, host.LOG_LIMIT - size)])
                            log.flush()
                            if size < host.LOG_LIMIT:
                                timeline.write(json.dumps({**base.stamp(), 'source': 'mixed-profiler-target-stdout',
                                    'chunk': data[:host.LOG_LIMIT-size].decode('utf-8', errors='replace')}) + '\n')
                                timeline.flush()
                            size += len(data)
                            if size > host.LOG_LIMIT: overflow.set()
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                except Exception: self.stream_error.set()
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            deadline = time.monotonic() + self.case_timeout
            admission_deadline = time.monotonic() + 30
            next_sample = next_copy = 0
            refresh = time.monotonic() + self.sudo_refresh_seconds
            with (self.output / 'process-timeline.jsonl').open('x') as timeline:
                while True:
                    self.tree.scan()
                    self.profiler.poll()
                    reason = self.stop_reason(deadline)
                    if reason or overflow.is_set():
                        reason = reason or 'console-overflow'
                        break
                    ready_path = self.directory / 'gate-ready.json'
                    if self.admitted is None and ready_path.exists():
                        # Same ownership/argv check as qualified CPU gate. CUDA
                        # profiler injection may already map libraries in Python;
                        # validate those explicitly instead of a GPU_LIB ban.
                        try: value = json.loads(gate.private_read(ready_path))
                        except json.JSONDecodeError: value = None
                        if value is not None:
                            pid = value.get('pid')
                            if type(pid) is not int or pid not in self.tree.members or not self.tree.alive(pid):
                                raise ValueError('gate PID is not an owned live descendant')
                            info = host.process_info(pid)
                            actual_argv = [os.fsdecode(x) for x in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if x]
                            if (value.get('nonce') != nonce or value.get('state') != 'waiting-before-exec' or
                                value.get('starttime') != info['starttime'] or
                                not host.same_process(self.tree.members[pid][0], info) or actual_argv != gate_argv or
                                (Path('/proc') / str(pid) / 'exe').resolve() != Path(sys.executable).resolve()):
                                raise ValueError('gate process identity mismatch')
                            self.admitted = value
                            self.summary['admitted_target'] = value
                            self.observe_mappings(pid)
                            self.verify_pins()
                            if base.sha256(gate_path) != self.gate_hash:
                                raise ValueError('copied gate changed')
                            if self.stop_reason(deadline) or not self.tree.alive(pid):
                                raise RuntimeError('fault/collector stop or target exit before release')
                            with (self.directory / 'gate-release').open('x') as release: release.write(nonce)
                            self.summary.update({'workload': 'running', 'workload_pid': pid, 'workload_start': base.stamp()})
                            self.save()
                    if self.admitted and self.tree.alive(self.admitted['pid']):
                        pid = self.admitted['pid']
                        if not self.target_verified:
                            self.target_verified = target_identity(pid, self.admitted, self.executable)
                            self.summary['exec_identity_verified'] = self.target_verified
                            if self.target_verified: next_sample = 0
                        if time.monotonic() >= next_sample:
                            sample = self.process_snapshot(pid)
                            sample['phase'] = 'frozen-executable' if self.target_verified else 'launch-gate'
                            sample['owned'] = [i for i, _ in self.tree.members.values()]
                            timeline.write(json.dumps(sample) + '\n'); timeline.flush()
                            self.observe_mappings(pid)
                            next_sample = time.monotonic() + 1
                    if time.monotonic() >= next_copy:
                        prefixes.capture()
                        _, excluded = host.safe_files(self.directory)
                        if excluded: raise RuntimeError('Nsight artifact size/type limit exceeded')
                        next_copy = time.monotonic() + 0.1
                    if time.monotonic() >= refresh:
                        self.required('sudo-refresh-' + str(time.monotonic_ns()), ['sudo', '-n', '-v'], seconds=5)
                        refresh = time.monotonic() + self.sudo_refresh_seconds
                    if self.profiler.returncode is not None and not self.tree.live_pids():
                        reason = 'processes-exited'; break
                    if self.admitted is None and time.monotonic() >= admission_deadline:
                        raise TimeoutError('gate admission deadline; executable not released')
                    time.sleep(0.05)
        except Exception as error:
            self.issue(str(error))
            reason = 'stopped-on-collector-failure'
        finally:
            self.summary['stop_reason'] = reason
            self.summary['stop_requested'] = base.stamp()
            try: prefixes.capture()
            except Exception as error: self.issue('prefix capture: ' + str(error))
            self.stop_owned()
            if reader:
                reader.join(timeout=2)
                if reader.is_alive(): self.issue('Nsight console reader did not finish')
            self.summary['profiler_returncode'] = self.profiler.poll()
            self.summary['target_exit_observed'] = bool(self.admitted and self.tree and not self.tree.alive(self.admitted['pid']))
            self.summary['workload_end'] = base.stamp()
            if reason != 'processes-exited': self.issue('Nsight session stopped: ' + str(reason))
            console = (self.directory / 'console.log').read_text(errors='replace') if (self.directory / 'console.log').exists() else ''
            clean = (reason == 'processes-exited' and self.target_verified and
                     self.summary['target_exit_observed'] and self.profiler.returncode == 0 and
                     'harness_status=ok' in console and not self.summary['issues'] and not self.fault.is_set())
            self.summary['workload'] = 'exited-zero' if clean else (reason if reason != 'processes-exited' else 'failed')
            self.summary['workload_success_basis'] = 'profiler status + target exit + harness marker; not a direct wait status'
            self.save()

    def capture(self):
        self.output = self.output.resolve()
        self.directory = self.output / 'nsys'
        self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        try:
            self.preflight()
            self.summary['preflight'] = 'passed'
            if not self.preflight_only: self.execute()
        except Exception as error:
            self.issue(str(error))
            print('Nsight capture: ' + str(error), file=sys.stderr, flush=True)
        finally:
            if self.profiler:
                try: self.stop_owned()
                except Exception as error: self.issue('owned cleanup incomplete: ' + str(error))
                try: self.postmortem()  # Preserve kernel/PCI evidence even if profiler cleanup fails.
                except Exception as error: self.issue('postmortem incomplete: ' + str(error))
                try:
                    if self.admitted and self.tree:
                        self.summary['target_exit_observed'] = not self.tree.alive(self.admitted['pid'])
                    self.summary['profiler_returncode'] = self.profiler.poll()
                    if self.summary.get('target_exit_observed') and not self.summary.get('surviving_owned_pids'):
                        self.summary['nsys_export'] = retention.export_report(self.nsys, self.directory)
                    else:
                        self.summary['nsys_export'] = {'status': 'deferred-owned-processes-not-confirmed-exited'}
                    if self.summary['nsys_export']['status'] != 'exported':
                        self.issue('Nsight report absent/incomplete; inspect retained prefixes and export status')
                    observed = self.summary.get('actual_target_mapped_hashes', {})
                    if not all(any(Path(p).name.startswith(lib) for p in observed) for lib in CORE_LIBS):
                        self.issue('actual target core-library mapping coverage incomplete')
                except Exception as error: self.issue('export/mapping review incomplete: ' + str(error))
            if self.tree: self.tree.close()
            self.finish()  # No workload_process assigned: never treat profiler as target.
        return self.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--qualification-archive', required=True)
    parser.add_argument('--nsys', default='/usr/local/cuda-13.2/bin/nsys')
    parser.add_argument('--case-timeout', type=int, default=600)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--run-frozen-gpu1', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.case_timeout <= 600: parser.error('timeout must be 1..600 seconds')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('the exact runner environment command is required even in preflight-only mode')
    os.umask(0o077)
    capture = NsightCapture(args.output, args.executable, command, args.case_timeout,
                           args.qualification_archive, args.nsys, args.preflight_only)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: capture.interrupted.set())
    return capture.capture()


if __name__ == '__main__': sys.exit(main())
