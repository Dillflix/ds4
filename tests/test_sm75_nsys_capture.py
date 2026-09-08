"""CPU-only integration checks; no real profiler, driver or workload executed."""
import importlib.util
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'speed-bench/capture-sm75-nsys.py'
spec = importlib.util.spec_from_file_location('nsys_capture', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exe = self.root / 'frozen'
        self.exe.write_bytes(b'not-an-executable')
        self.env = {'LD_LIBRARY_PATH': '/usr/local/cuda/lib64:'}
        self.command = ['env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES=1',
            'DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_NO_ROW_OWNED=1',
            'DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_CALLS=1024',
            'DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_BATCH=10',
            'B_TIMING_ROUNDS=7', 'B_TIMING_REPEATS=10', 'B_TIMING_WARMUPS=3', str(self.exe)]
        self.c = m.NsightCapture(self.root / 'out', self.exe, self.command, 600,
                                 self.root / 'receipt', self.root / 'nsys')

    def test_exact_environment_and_physical_uuid(self):
        argv, env = m.exact_environment(self.command, self.exe, self.env)
        self.assertEqual(argv, [str(self.exe)])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], m.base.GPU1_UUID)
        self.assertEqual(env['B_TIMING_REPEATS'], '10')

    def test_wrapper_selectors_do_not_reach_profiler(self):
        env = dict(self.env, NSYS_CAPTURE='1', NSYS_QUALIFICATION_ARCHIVE='archive')
        _, child = m.exact_environment(self.command, self.exe, env)
        self.assertNotIn('NSYS_CAPTURE', child)
        self.assertNotIn('NSYS_QUALIFICATION_ARCHIVE', child)

    def test_changed_reproducer_settings_rejected(self):
        for i in range(1, len(self.command)-1):
            command = self.command.copy()
            key = command[i].split('=')[0]
            command[i] = key + '=wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.exact_environment(command, self.exe, self.env)

    def test_alternate_executable_or_arguments_rejected(self):
        for command in (self.command + ['--short'], self.command[:-1] + ['other']):
            with self.assertRaises(ValueError): m.exact_environment(command, self.exe, self.env)

    def test_runtime_controls_must_match_including_unset(self):
        for key in m.base.RUNTIME_CONTROLS:
            if key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'): continue
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.exact_environment(self.command, self.exe, dict(self.env, **{key: ''}))

    def test_mixed_instrumentation_rejected(self):
        for key in ('LD_PRELOAD', 'LD_AUDIT', 'CUDA_INJECTION64_PATH', 'CUDA_INJECTION32_PATH',
                    'NVTX_INJECTION64_PATH', 'NSYS_TEST', 'CUPTI_TEST', 'CUBLAS_LOGINFO_DBG',
                    'DS4_RUNTIME_TRACE_LOG', 'NV_COMPUTE_SANITIZER_TEST'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.exact_environment(self.command, self.exe, dict(self.env, **{key: 'enabled'}))

    def test_extra_ds4_selector_rejected(self):
        with self.assertRaises(ValueError):
            m.exact_environment(self.command, self.exe, dict(self.env, DS4_ALGO='104'))

    def test_receipt_hash_mismatch_rejected_without_parsing(self):
        receipt = self.root / 'receipt'
        receipt.write_bytes(b'wrong')
        with patch.object(m.tarfile, 'open') as opened:
            with self.assertRaisesRegex(ValueError, 'reviewed'): m.receipt(receipt)
            opened.assert_not_called()

    def test_runtime_pins_detect_changed_library(self):
        self.c.pinned = {'lib': 'expected'}
        with patch.object(m.base, 'sha256', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'library changed'): self.c.verify_pins()

    def test_frozen_binary_rechecked_after_library_pins(self):
        self.c.pinned = {}
        with patch.object(m.base, 'sha256', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'executable changed'): self.c.verify_pins()

    def test_cuda_options_do_not_enable_events_metrics_or_replay(self):
        self.assertIn('--cuda-event-trace=false', m.CUDA_OPTIONS)
        self.assertIn('--cuda-trace-scope=process-tree', m.CUDA_OPTIONS)
        self.assertIn('--trace=cuda-sw,cublas-verbose', m.CUDA_OPTIONS)
        self.assertIn('--kill=none', m.host.HOST_OPTIONS)
        self.assertNotIn('--cuda-event-trace=true', m.CUDA_OPTIONS)

    def test_stop_before_profiler_launch_on_fault(self):
        self.c.fault.set()
        with patch.object(m.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'before profiler launch'): self.c.execute()
            launch.assert_not_called()

    def test_journal_failure_requests_stop(self):
        self.c.journal_process = Mock(poll=Mock(return_value=1))
        self.c.journal_reader = Mock(is_alive=Mock(return_value=True))
        self.assertEqual(self.c.stop_reason(float('inf')), 'stopped-on-collector-failure')

    def test_dead_reader_requests_stop(self):
        self.c.journal_process = Mock(poll=Mock(return_value=None))
        self.c.journal_reader = Mock(is_alive=Mock(return_value=False))
        self.assertEqual(self.c.stop_reason(float('inf')), 'stopped-on-collector-failure')

    def test_deadline_requests_stop(self):
        self.c.journal_process = Mock(poll=Mock(return_value=None))
        self.c.journal_reader = Mock(is_alive=Mock(return_value=True))
        self.assertEqual(self.c.stop_reason(-1), 'timeout')

    def test_stop_signals_verified_target_before_tree(self):
        self.c.directory.mkdir(parents=True)
        tree = Mock(alive=Mock(side_effect=[True, False, False, False]), stop=Mock(return_value=[]),
                    live_pids=Mock(return_value=[]), cleanup_issues=[])
        self.c.tree, self.c.profiler, self.c.admitted = tree, Mock(poll=Mock(return_value=0)), {'pid': 11}
        with patch.object(m.signal, 'SIGKILL', getattr(signal, 'SIGKILL', 9), create=True):
            self.c.stop_owned()
        tree.send.assert_called_once_with(11, signal.SIGTERM)
        self.assertLess([x[0] for x in tree.method_calls].index('send'),
                        [x[0] for x in tree.method_calls].index('stop'))
        tree.stop.assert_called_once()

    def test_survivors_make_collection_partial(self):
        self.c.directory.mkdir(parents=True)
        self.c.tree = Mock(stop=Mock(return_value=[11]), cleanup_issues=[])
        self.c.profiler = Mock(poll=Mock(return_value=0))
        self.c.stop_owned()
        self.assertIn('survived', self.c.summary['issues'][0])

    def test_actual_mapping_mismatch_rejected(self):
        path = next(iter(m.RUNTIME))
        self.c.bundle_inventory = {}
        with patch.object(m, 'mapped_files', return_value=[path]), patch.object(m.base, 'sha256', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'mismatch'): self.c.observe_mappings(11)

    def test_unreviewed_driver_mapping_rejected(self):
        self.c.bundle_inventory = {}
        with patch.object(m, 'mapped_files', return_value=['/unexpected/libcuda.so.99']), \
             patch.object(m.base, 'sha256', return_value='hash'):
            with self.assertRaisesRegex(ValueError, 'unreviewed'): self.c.observe_mappings(11)

    def test_gate_libraries_do_not_count_as_post_exec_target_coverage(self):
        path, digest = next(iter(m.RUNTIME.items()))
        self.c.bundle_inventory = {}
        with patch.object(m, 'mapped_files', return_value=[path]), patch.object(m.base, 'sha256', return_value=digest):
            self.c.observe_mappings(11)
            self.assertNotIn('actual_target_mapped_hashes', self.c.summary)
            self.assertEqual(self.c.summary['gate_mapped_hashes'][path], digest)
            self.c.target_verified = True
            fake = Mock(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5)
            with patch.object(Path, 'stat', return_value=fake):
                self.c.observe_mappings(11)
                self.c.finish_workers()
            self.assertEqual(self.c.summary['actual_target_mapped_hashes'][path], digest)

    def test_transient_mapping_loss_not_silently_qualified(self):
        self.c.bundle_inventory = {}
        with patch.object(m, 'mapped_files', return_value=['/opt/nsight/component (deleted)']):
            self.c.observe_mappings(11)
        self.assertIn('vanished', self.c.summary['issues'][0])

    def test_preflight_only_never_enters_execute_or_postmortem(self):
        self.c.preflight_only = True
        with patch.object(self.c, 'preflight'), patch.object(self.c, 'execute') as execute, \
             patch.object(self.c, 'postmortem') as post:
            self.assertEqual(self.c.capture(), 0)
            execute.assert_not_called(); post.assert_not_called()

    def test_preflight_failure_never_enters_execute(self):
        with patch.object(self.c, 'preflight', side_effect=ValueError('wrong stack')), \
             patch.object(self.c, 'execute') as execute:
            self.assertEqual(self.c.capture(), 2)
            execute.assert_not_called()

    def test_running_driver_change_rejected(self):
        with patch.object(m.base.Capture, 'preflight'), \
             patch.object(Path, 'read_text', return_value='NVRM version 595.84'):
            with self.assertRaisesRegex(ValueError, 'running NVIDIA module'): self.c.preflight()

    def test_cuda_alias_change_rejected(self):
        with patch.object(m.base.Capture, 'preflight'), \
             patch.object(Path, 'read_text', return_value='NVRM version 595.91.07'), \
             patch.object(Path, 'resolve', return_value=Path('/different/cuda')):
            with self.assertRaisesRegex(ValueError, 'alias changed'): self.c.preflight()

    def test_profiler_cleanup_error_does_not_skip_kernel_pci_postmortem(self):
        self.c.profiler = Mock(poll=Mock(return_value=1))
        with patch.object(self.c, 'preflight'), patch.object(self.c, 'execute'), \
             patch.object(self.c, 'stop_owned', side_effect=RuntimeError('cleanup problem')), \
             patch.object(self.c, 'postmortem') as post:
            self.c.capture()
            post.assert_called_once()
        self.assertTrue(any('cleanup incomplete' in issue for issue in self.c.summary['issues']))

    def test_profiler_zero_does_not_fill_target_returncode(self):
        self.c.profiler = Mock(poll=Mock(return_value=0))
        self.c.observe_final_workload_status()
        self.assertIsNone(self.c.summary['workload_returncode'])
        self.assertIsNone(self.c.workload_process)

    def test_cli_requires_explicit_mode(self):
        with patch.object(m.sys, 'argv', ['capture', '--output', 'x', '--executable', 'y',
                                         '--qualification-archive', 'z']), patch.object(m, 'NsightCapture') as capture:
            with self.assertRaises(SystemExit): m.main()
            capture.assert_not_called()

    def simulated_execute(self, *, stop_before_release=False, profiler_code=0, marker=True):
        c = self.c
        c.output.mkdir()
        c.directory.mkdir()
        c.environment = self.env
        c.gate_hash = 'hash'
        gatepath = c.directory / 'sm75-nsys-launch-gate.py'
        gatepath.write_text('not executed')
        c.journal_process = Mock(poll=Mock(return_value=None))
        c.journal_reader = Mock(is_alive=Mock(return_value=True))
        process = Mock(pid=10, returncode=None,
                       stdout=io.BytesIO(b'harness_status=ok\n' if marker else b'no success\n'))
        polls = [0]
        def poll():
            polls[0] += 1
            if polls[0] > 1: process.returncode = profiler_code
            return process.returncode
        process.poll.side_effect = poll
        info = {'pid': 42, 'ppid': 10, 'starttime': 7, 'uid': 1}
        tree = Mock(members={42: (info, 55)}, cleanup_issues=[])
        tree.alive.side_effect = lambda pid: process.returncode is None
        tree.live_pids.side_effect = lambda: [10, 42] if process.returncode is None else []
        tree.stop.return_value = []
        def scan():
            config = json.loads((c.directory / 'gate-config.json').read_text())
            value = dict(info, nonce=config['nonce'], state='waiting-before-exec')
            (c.directory / 'gate-ready.json').write_text(json.dumps(value))
        tree.scan.side_effect = scan
        gateargv = [m.sys.executable, '-I', str(gatepath), str(c.directory / 'gate-config.json')]
        real_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            if str(path).replace('\\', '/') == '/proc/42/exe': return real_resolve(Path(m.sys.executable))
            return real_resolve(path, *args, **kwargs)
        with ExitStack() as stack:
            for target, attr, value in (
                (c, 'verify_pins', Mock()), (c, 'observe_mappings', Mock()),
                (c, 'process_snapshot', Mock(return_value={'pid': 42})),
                (m.base, 'sha256', Mock(return_value='hash')),
                (m.subprocess, 'Popen', Mock(return_value=process)),
                (m.host, 'OwnedTree', Mock(return_value=tree)),
                (m.host, 'process_info', Mock(return_value=info)),
                (m.host, 'safe_files', Mock(return_value=([], []))),
                (m.retention, 'PrefixStore', Mock(return_value=Mock())),
                (m.gate, 'private_read', lambda p: Path(p).read_text()),
                (m, 'target_identity', Mock(return_value=True)),
                (Path, 'read_bytes', Mock(return_value=b'\0'.join(os.fsencode(x) for x in gateargv)+b'\0')),
                (Path, 'resolve', resolve),
                (m.sys, 'stdout', Mock(buffer=io.BytesIO())),
            ):
                stack.enter_context(patch.object(target, attr, value))
            stack.enter_context(patch.object(m.os, 'O_NOFOLLOW', getattr(os, 'O_NOFOLLOW', 0), create=True))
            stack.enter_context(patch.object(m.signal, 'SIGKILL', getattr(signal, 'SIGKILL', 9), create=True))
            if stop_before_release:
                stack.enter_context(patch.object(c, 'stop_reason', side_effect=[None, None, 'stopped-on-kernel-fault']))
            c.execute()
            c.finish_workers()
        return c.summary, tree

    def test_mock_complete_session_preserves_target_and_profiler_separation(self):
        summary, tree = self.simulated_execute()
        self.assertEqual(summary['workload'], 'exited-zero')
        self.assertEqual(summary['workload_pid'], 42)
        self.assertEqual(summary['profiler_pid'], 10)
        self.assertEqual(summary['profiler_returncode'], 0)
        self.assertIsNone(summary['workload_returncode'])
        self.assertTrue(summary['exec_identity_verified'])
        self.assertTrue((self.c.output / 'application-timeline.jsonl').is_file())

    def test_fault_before_release_never_opens_barrier(self):
        summary, tree = self.simulated_execute(stop_before_release=True)
        self.assertFalse((self.c.directory / 'gate-release').exists())
        self.assertNotEqual(summary['workload'], 'exited-zero')
        tree.stop.assert_called_once()

    def test_profiler_zero_without_harness_marker_fails(self):
        summary, _ = self.simulated_execute(marker=False)
        self.assertEqual(summary['workload'], 'failed')

    def test_profiler_nonzero_with_harness_marker_fails(self):
        summary, _ = self.simulated_execute(profiler_code=137)
        self.assertEqual(summary['workload'], 'failed')
        self.assertIsNone(summary['workload_returncode'])

    def test_blocked_hash_does_not_block_fault_checks_or_prefix_retention(self):
        entered, release, copied = threading.Event(), threading.Event(), threading.Event()
        path = self.root / 'libcublas-test'
        path.write_bytes(b'data')
        def digest(_):
            entered.set()
            if not release.wait(2): raise RuntimeError('test release timeout')
            return 'verified'
        with patch.object(m.base, 'sha256', side_effect=digest), \
             patch.object(m.host, 'safe_files', return_value=([], [])):
            hasher = m.MappingHasher()
            copier = m.PrefixCollector(Mock(capture=Mock(side_effect=copied.set)), self.root)
            try:
                self.c.mapping_hasher, self.c.prefix_collector = hasher, copier
                hasher.submit('actual_target_mapped_hashes', str(path), 'verified')
                self.assertTrue(entered.wait(1))
                self.assertTrue(copied.wait(1))
                self.c.fault.set()
                self.assertEqual(self.c.stop_reason(float('inf')), 'stopped-on-kernel-fault')
                self.assertNotIn('actual_target_mapped_hashes', self.c.summary)
            finally:
                release.set()
                self.c.finish_workers()
            self.assertEqual(self.c.summary['actual_target_mapped_hashes'][str(path)], 'verified')
            self.assertTrue(self.c.summary['mapping_hasher']['stopped'])

    def test_target_mapping_submits_without_synchronous_hash(self):
        path = next(iter(m.RUNTIME))
        self.c.target_verified = True
        self.c.bundle_inventory = {}
        self.c.mapping_hasher = Mock()
        with patch.object(m, 'mapped_files', return_value=[path]), patch.object(m.base, 'sha256') as digest:
            self.c.observe_mappings(11)
            digest.assert_not_called()
        self.c.mapping_hasher.submit.assert_called_once_with('actual_target_mapped_hashes', path, m.RUNTIME[path])
        self.assertNotIn('actual_target_mapped_hashes', self.c.summary)

    def test_hash_error_is_stop_reason_not_successful_coverage(self):
        path = self.root / 'lib'
        path.write_bytes(b'data')
        with patch.object(m.base, 'sha256', return_value='wrong'):
            self.c.mapping_hasher = m.MappingHasher()
            self.c.mapping_hasher.submit('actual_target_mapped_hashes', str(path), 'expected')
            self.assertTrue(self.c.mapping_hasher.failed.wait(1))
            self.assertEqual(self.c.stop_reason(float('inf')), 'stopped-on-collector-failure')
            self.c.finish_workers()
        self.assertNotIn('actual_target_mapped_hashes', self.c.summary)
        self.assertTrue(any('mismatch' in issue for issue in self.c.summary['issues']))

    def test_changed_file_during_worker_hash_rejected(self):
        path = self.root / 'lib'
        path.write_bytes(b'data')
        def digest(_):
            path.write_bytes(b'changed size')
            return 'expected'
        with patch.object(m.base, 'sha256', side_effect=digest):
            self.c.mapping_hasher = m.MappingHasher()
            self.c.mapping_hasher.submit('actual_target_mapped_hashes', str(path), 'expected')
            self.assertTrue(self.c.mapping_hasher.failed.wait(1))
            self.c.finish_workers()
        self.assertTrue(any('changed during' in issue for issue in self.c.summary['issues']))

    def test_worker_job_limit(self):
        worker = m.MappingHasher()
        try:
            worker.submitted = 256
            with self.assertRaisesRegex(ValueError, 'job limit'): worker.submit('phase', 'path', None)
        finally: worker.stop()

    def test_pending_hashes_mark_evidence_incomplete(self):
        self.c.mapping_hasher = Mock(submitted=5, stop=Mock(return_value=False), results=m.queue.Queue())
        self.c.finish_workers()
        self.assertIn('incomplete', self.c.summary['issues'][0])

    def test_prefix_failure_requests_stop(self):
        worker = m.PrefixCollector(Mock(capture=Mock(side_effect=OSError('disk error'))), self.root)
        try:
            self.assertTrue(worker.failed.wait(1))
            self.c.prefix_collector = worker
            self.assertEqual(self.c.stop_reason(float('inf')), 'stopped-on-collector-failure')
            self.c.finish_workers()
        finally: worker.stop()
        self.assertIn('disk error', self.c.summary['issues'][0])

    def test_prefix_capture_does_final_copy_after_stop_request(self):
        seen = threading.Event()
        values, state = [], [1]
        def capture():
            values.append(state[0])
            seen.set()
        with patch.object(m.host, 'safe_files', return_value=([], [])):
            worker = m.PrefixCollector(Mock(capture=capture), self.root)
            try:
                self.assertTrue(seen.wait(1))
                state[0] = 2
            finally:
                self.assertTrue(worker.stop())
        self.assertEqual(values[-1], 2)

    def test_blocked_prefix_copy_has_bounded_join_and_does_not_block_fault_check(self):
        entered, release = threading.Event(), threading.Event()
        def capture():
            entered.set()
            release.wait(2)
        with patch.object(m.host, 'safe_files', return_value=([], [])):
            worker = m.PrefixCollector(Mock(capture=capture), self.root)
            try:
                self.assertTrue(entered.wait(1))
                self.c.prefix_collector = worker
                self.c.fault.set()
                self.assertEqual(self.c.stop_reason(float('inf')), 'stopped-on-kernel-fault')
                self.assertFalse(worker.stop(seconds=0.01))
            finally:
                release.set()
                self.assertTrue(worker.stop())

    def simulated_shutdown(self, target_alive=False, profiler_exit_at=None):
        self.c.directory.mkdir(parents=True)
        elapsed = [0.0]
        self.c.admitted = {'pid': 11}
        self.c.tree = Mock(alive=Mock(return_value=target_alive),
                           live_pids=Mock(return_value=[11] if target_alive else []),
                           stop=Mock(return_value=[11] if target_alive else []), cleanup_issues=[])
        self.c.profiler = Mock(poll=Mock(side_effect=lambda:
            0 if profiler_exit_at is not None and elapsed[0] >= profiler_exit_at else None))
        with patch.object(m.time, 'monotonic', side_effect=lambda: elapsed[0]), \
             patch.object(m.time, 'sleep', side_effect=lambda dt: elapsed.__setitem__(0, elapsed[0]+dt)), \
             patch.object(m.signal, 'SIGKILL', getattr(signal, 'SIGKILL', 9), create=True):
            self.c.stop_owned()
            before = elapsed[0]
            self.c.stop_owned()
            self.assertEqual(before, elapsed[0])  # No second grace period.
        return elapsed[0], self.c.summary['shutdown_timeline']

    def test_profiler_may_finalize_after_old_five_second_limit(self):
        elapsed, events = self.simulated_shutdown(profiler_exit_at=8)
        self.assertGreaterEqual(elapsed, 8)
        self.assertLess(elapsed, 9)
        end = next(x for x in events if x['event'] == 'profiler-finalization-end')
        self.assertFalse(end['deadline_expired'])
        self.assertEqual(end['profiler_returncode'], 0)

    def test_profiler_grace_has_hard_cap_and_is_not_restarted(self):
        elapsed, events = self.simulated_shutdown()
        self.assertGreaterEqual(elapsed, m.FINALIZE_SECONDS)
        self.assertLess(elapsed, m.FINALIZE_SECONDS + 0.1)
        self.assertTrue(next(x for x in events if x['event'] == 'profiler-finalization-end')['deadline_expired'])
        self.c.tree.stop.assert_called_once()

    def test_profiler_children_can_finalize_after_launcher_exits(self):
        self.c.directory.mkdir(parents=True)
        self.c.admitted = {'pid': 11}
        elapsed = [0.0]
        self.c.tree = Mock(alive=Mock(return_value=False), stop=Mock(return_value=[]), cleanup_issues=[],
            live_pids=Mock(side_effect=lambda: [22] if elapsed[0] < 8 else []))
        self.c.profiler = Mock(poll=Mock(return_value=0))
        with patch.object(m.time, 'monotonic', side_effect=lambda: elapsed[0]), \
             patch.object(m.time, 'sleep', side_effect=lambda dt: elapsed.__setitem__(0, elapsed[0]+dt)), \
             patch.object(m.signal, 'SIGKILL', getattr(signal, 'SIGKILL', 9), create=True):
            self.c.stop_owned()
        self.assertGreaterEqual(elapsed[0], 8)
        self.assertLess(elapsed[0], 9)

    def test_stuck_target_gets_no_profiler_grace(self):
        elapsed, events = self.simulated_shutdown(target_alive=True)
        self.assertLess(elapsed, 2.1)
        self.assertNotIn('profiler-finalization-start', [x['event'] for x in events])

    def test_timeline_write_failure_cannot_skip_owned_stop(self):
        self.c.tree = Mock(stop=Mock(return_value=[]), cleanup_issues=[])
        self.c.profiler = Mock(poll=Mock(return_value=0))
        with patch.object(Path, 'open', side_effect=OSError('disk full')):
            self.c.stop_owned()
        self.c.tree.stop.assert_called_once()
        self.assertTrue(any('timeline write failed' in x for x in self.c.summary['issues']))


if __name__ == '__main__': unittest.main()
