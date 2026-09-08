#!/usr/bin/env python3
"""CPU-only trace-mode wiring/security tests; fake ELF is never loaded/executed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / 'speed-bench/capture-sm75-gpu1-failure.py'
spec = importlib.util.spec_from_file_location('runtime_capture_tests', path)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def command(executable):
    return ['env', '-u', 'CUDA_LAUNCH_BLOCKING', '-u', 'DS4_OTHER',
            'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES=1',
            'DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_NO_ROW_OWNED=1',
            'DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_CALLS=1024',
            'DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_BATCH=10',
            'B_TIMING_ROUNDS=7', 'B_TIMING_REPEATS=10', 'B_TIMING_WARMUPS=3', str(executable)]


class ChildEnvironment(unittest.TestCase):
    def setUp(self):
        self.executable = Path(sys.executable).resolve()
        self.command = command(self.executable)

    def test_direct_owned_elf_not_env_wrapper(self):
        inherited = {'DS4_OTHER': '1', 'CUDA_LAUNCH_BLOCKING': '1', 'LD_LIBRARY_PATH': '/kept'}
        argv, env = capture.trace_child_command(self.command, self.executable, inherited)
        self.assertEqual(argv, [str(self.executable)])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], capture.GPU1_UUID)
        self.assertEqual(env['LD_LIBRARY_PATH'], '/kept')
        self.assertNotIn('DS4_OTHER', env)
        self.assertNotIn('CUDA_LAUNCH_BLOCKING', env)
        self.assertIn('DS4_OTHER', inherited)  # no parent mutation

    def test_reduced_call_count_refused(self):
        self.command = [v.replace('CALLS=1024', 'CALLS=1') for v in self.command]
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            capture.trace_child_command(self.command, self.executable, {})

    def test_extra_ds4_selector_refused(self):
        with self.assertRaisesRegex(ValueError, 'extra DS4'):
            capture.trace_child_command(self.command, self.executable, {'DS4_NEW_EXPERIMENT': '1'})

    def test_profiler_shell_wrapper_refused(self):
        for prefix in ['nsys', 'bash', 'sh', 'different-env']:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                capture.trace_child_command([prefix, *self.command[1:]], self.executable, {})

    def test_extra_elf_argument_refused(self):
        with self.assertRaisesRegex(ValueError, 'without arguments'):
            capture.trace_child_command(self.command + ['--small'], self.executable, {})

    def test_substituted_elf_refused(self):
        with self.assertRaises(ValueError):
            capture.trace_child_command([*self.command[:-1], 'other-binary'], self.executable, {})

    def test_injected_controls_refused(self):
        for setting in ['LD_PRELOAD=x.so', 'LD_AUDIT=x.so', 'CUPTI_INJECTION_PATH=x', 'DS4_RUNTIME_TRACE_LOG=/tmp/x',
                        'CUBLAS_LOGINFO_DBG=1', 'CUDA_INJECTION64_PATH=x.so',
                        'CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1']:
            with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, 'instrumentation'):
                capture.trace_child_command([*self.command[:-1], setting, self.command[-1]], self.executable, {})

    def test_malformed_unset_refused(self):
        with self.assertRaisesRegex(ValueError, 'unset'):
            capture.trace_child_command(['env', '-u', '=bad'], self.executable, {})

    def test_linux_loader_delimiters_rejected(self):
        self.assertTrue(capture.valid_preload_path('/tmp/trace.so'))
        for bad in ['/tmp/a b.so', '/tmp/a:b.so', '/tmp/a\nb.so', '/tmp/a\tb.so']:
            self.assertFalse(capture.valid_preload_path(bad))


class LibraryPreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='runtime-contract-cpu-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lib = self.root/'fixture.so'
        # Enough to test file admission, explicitly not a loadable shared object.
        self.lib.write_bytes(b'\x7fELF\x02\x01' + b'\0'*10 + b'\x03\x00\x3e\x00' + b'\0'*44)
        self.lib.chmod(0o500)
        self.digest = hashlib.sha256(self.lib.read_bytes()).hexdigest()
        self.output = self.root/'capture'
        self.output.mkdir()
        self.obj = capture.Capture(self.output, sys.executable, command(sys.executable), 600,
                                   runtime_trace_library=self.lib, runtime_trace_sha256=self.digest)

    def prepare(self):
        with patch.dict(capture.os.environ, {}, clear=True), \
             patch.object(capture.os, 'getuid', return_value=self.lib.stat().st_uid, create=True), \
             patch.object(capture, 'valid_preload_path', return_value=True):
            # Windows drive colons in these CPU-only fixture paths are not a
            # Linux loader path. Its actual delimiter rules are tested above.
            self.obj.prepare_runtime_trace()

    def test_explicit_trace_copied_and_only_child_environment_changed(self):
        with patch.dict(capture.os.environ, {}, clear=True):
            self.prepare()
            self.assertNotIn('LD_PRELOAD', capture.os.environ)
            self.assertNotIn(capture.TRACE_ENV, capture.os.environ)
        info = self.obj.summary['runtime_trace']
        self.assertEqual(info['library_sha256'], self.digest)
        self.assertEqual(hashlib.sha256(Path(info['library_copy']).read_bytes()).hexdigest(), self.digest)
        self.assertEqual(self.obj.trace_command, [str(Path(sys.executable).resolve())])
        self.assertEqual(self.obj.trace_environment['LD_PRELOAD'], info['library_copy'])
        self.assertFalse(Path(info['log']).exists())  # native logger creates exclusively
        self.assertEqual(self.obj.summary['execution_mode'], 'instrumented-runtime-contract')

    def test_no_opt_in_leaves_original_execution(self):
        self.obj.runtime_trace_library = None
        self.obj.runtime_trace_sha256 = None
        self.prepare()
        self.assertIsNone(self.obj.trace_command)
        self.assertFalse((self.output/'runtime-trace').exists())

    def test_hash_without_library_refused(self):
        self.obj.runtime_trace_library = None
        with self.assertRaisesRegex(ValueError, 'explicit library'):
            self.prepare()

    def test_missing_hash_refused(self):
        self.obj.runtime_trace_sha256 = None
        with self.assertRaisesRegex(ValueError, 'explicit lowercase'):
            self.prepare()

    def test_mismatched_hash_refused(self):
        self.obj.runtime_trace_sha256 = '1'*64
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            self.prepare()
        self.assertFalse((self.output/'runtime-trace').exists())

    def test_wrong_elf_architecture_refused(self):
        self.lib.chmod(0o600)
        self.lib.write_bytes(b'notELF'+b'\0'*58)
        self.lib.chmod(0o500)
        with self.assertRaisesRegex(ValueError, 'ELF64'):
            self.prepare()

    def test_existing_trace_directory_refused(self):
        (self.output/'runtime-trace').mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_preflight_only_no_elf_or_library_execution(self):
        self.obj.preflight_only = True
        self.obj.output = self.root/'new-output'
        self.obj.preflight = self.prepare
        self.obj.execute = Mock()
        self.obj.postmortem = Mock()
        self.assertEqual(self.obj.capture(), 0)
        self.obj.execute.assert_not_called()
        self.obj.postmortem.assert_not_called()

    def test_missing_trace_after_child_is_partial_not_success(self):
        self.prepare()
        self.obj.workload_process = Mock()
        self.obj.finalize_runtime_trace()
        self.assertIn('trace missing', self.obj.summary['issues'][0])

    def test_uninstrumented_capture_does_not_analyze_trace(self):
        self.obj.finalize_runtime_trace()
        self.assertEqual(self.obj.summary['issues'], [])

    def test_partial_log_is_snapshotted_even_with_surviving_child(self):
        self.prepare()
        raw = self.output/'runtime-trace/runtime-contract.jsonl'
        raw.write_bytes(b'{"partial":')
        self.obj.workload_process = Mock(pid=123)
        self.obj.workload_process.poll.return_value = None
        self.obj.finalize_runtime_trace()
        snapshot = raw.with_name('runtime-contract.snapshot.jsonl')
        self.assertEqual(snapshot.read_bytes(), b'{"partial":')
        self.assertTrue(self.obj.summary['runtime_trace']['snapshot_of_live_child'])
        self.assertEqual(self.obj.summary['runtime_trace']['trace_sha256'], hashlib.sha256(snapshot.read_bytes()).hexdigest())
        self.assertTrue(self.obj.summary['issues'])
        raw.write_bytes(b'{"different":true}\n')
        self.assertEqual(snapshot.read_bytes(), b'{"partial":')

    def test_complete_but_no_api_trace_is_not_accepted(self):
        self.prepare()
        raw = self.output/'runtime-trace/runtime-contract.jsonl'
        rows=[]
        for seq,api in [(1,'trace_start'),(2,'trace_end')]:
            rows.append(json.dumps({'schema':'ds4-runtime-contract-v1','seq':seq,'call_id':0,
                'event':'meta','api':api,'pid':123,'tid':123,'monotonic_ns':seq,
                'realtime_ns':1780000000000000000+seq,'result':{}}))
        raw.write_text('\n'.join(rows)+'\n')
        self.obj.workload_process = Mock(pid=123)
        self.obj.workload_process.poll.return_value = 0
        self.obj.finalize_runtime_trace()
        self.assertTrue(any('incomplete' in x for x in self.obj.summary['issues']))

    def test_another_pid_trace_is_not_owned_workload(self):
        self.prepare()
        raw = self.output/'runtime-trace/runtime-contract.jsonl'
        raw.write_text(json.dumps({'schema':'ds4-runtime-contract-v1','seq':1,'call_id':0,
            'event':'meta','api':'trace_start','pid':456,'tid':456,'monotonic_ns':1,
            'realtime_ns':1780000000000000000,'result':{}})+'\n')
        self.obj.workload_process = Mock(pid=123)
        self.obj.workload_process.poll.return_value = 0
        self.obj.finalize_runtime_trace()
        self.assertTrue(any('PID' in x for x in self.obj.summary['issues']))

    def test_prelude_transition_does_not_qualify_postburnin_capture(self):
        self.prepare()
        raw = self.output/'runtime-trace/runtime-contract.jsonl'
        raw.write_bytes(b'{"snapshot_fixture":true}\n')
        self.obj.workload_process = Mock(pid=123)
        self.obj.workload_process.poll.return_value = 0
        report = {'process_ids': [123], 'required_capture_present': True,
                  'trace_complete': True, 'violations': [],
                  'expected_transition': {'expected_transition_required_present': True,
                                          'postburnin_transition_required_present': False}}
        fake = Mock(analyze_file=Mock(return_value=report))
        with patch.object(capture.importlib.util, 'spec_from_file_location', return_value=Mock()), \
             patch.object(capture.importlib.util, 'module_from_spec', return_value=fake):
            self.obj.finalize_runtime_trace()
        self.assertTrue(any('post-burn-in' in x for x in self.obj.summary['issues']))

    def test_qualified_analysis_has_no_collection_issue_but_no_safety_claim(self):
        self.prepare()
        raw = self.output/'runtime-trace/runtime-contract.jsonl'
        raw.write_bytes(b'{"snapshot_fixture":true}\n')
        self.obj.workload_process = Mock(pid=123)
        self.obj.workload_process.poll.return_value = 0
        report = {'process_ids': [123], 'required_capture_present': True,
                  'trace_complete': True, 'violations': [],
                  'status': 'observed-contract-with-gaps',
                  'expected_transition': {'expected_transition_required_present': True,
                                          'postburnin_transition_required_present': True}}
        fake = Mock(analyze_file=Mock(return_value=report))
        with patch.object(capture.importlib.util, 'spec_from_file_location', return_value=Mock()), \
             patch.object(capture.importlib.util, 'module_from_spec', return_value=fake):
            self.obj.finalize_runtime_trace()
        self.assertEqual(self.obj.summary['issues'], [])
        self.assertEqual(self.obj.summary['runtime_trace']['analysis_status'], 'observed-contract-with-gaps')


if __name__ == '__main__':
    unittest.main()
