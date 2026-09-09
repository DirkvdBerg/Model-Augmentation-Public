"""The stall diagnosis itself must work, or a stalled run tells us nothing.

Category: verification of the DIAGNOSTIC TOOL. Not a claim about the gantry.

Every check here exists because the corresponding failure is silent, and two of them are
failures this file's subject actually had:

  1. `faulthandler` has ONE global timer. A per-operation watchdog nested inside a whole-run
     heartbeat cancels the heartbeat when it exits, so the long-range dump stops after the first
     timed block. Measured: a 30-minute heartbeat produced an EMPTY dump file across a 40-minute
     run. An empty file reads as "nothing was wrong".
  2. Keying CUDA bookkeeping on `torch.cuda.is_available()` rather than on the target device
     makes every timed block initialise and synchronise CUDA during a CPU run. The symptom was
     itself diagnostic: each operation reported ~105% CPU while the process accrued ~2%, so the
     time was going BETWEEN the timed blocks.
  3. A watchdog that does not fire, or fires without naming the call, is worse than none: it
     converts "we do not know" into "we checked".

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/test_diag.py
"""
__project_origin__ = "added"

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch                                                          # noqa: E402
from gantry import diag                                               # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


def test_watchdog_fires_and_names_the_call():
    led = diag.OpLedger('t', echo=False)
    path = os.path.join(tempfile.mkdtemp(), 'w.log')
    fired = False
    try:
        with diag.timed('deliberately slow', ledger=led, timeout=1, dump_to=path, hard=True):
            time.sleep(2.2)
    except diag.OperationTimeout as e:
        fired = True
        msg = str(e)
    check('an overrunning operation raises OperationTimeout', fired)
    check('the message reports the CPU fraction, which is what says blocked vs slow',
          fired and 'CPU' in msg, msg[:80] if fired else '')
    body = open(path).read() if os.path.exists(path) else ''
    check('a stack dump is written and names this file',
          'Timeout (' in body and 'test_diag' in body,
          f'{len(body)} bytes of stack dump')
    check('the ledger records the row even though the body overran',
          len(led.rows) == 1 and led.rows[0]['label'] == 'deliberately slow')
    check('cpu_fraction is near zero for a sleeping op',
          led.rows[0]['cpu_fraction'] < 0.5, f"{led.rows[0]['cpu_fraction']:.3f}")


def test_incomplete_names_the_running_op():
    """The single fact four stalled attempts could not produce: WHICH call was running."""
    led = diag.OpLedger('t', echo=False)
    seen = {}
    try:
        with diag.timed('the one that never returns', ledger=led):
            seen['during'] = dict(led.incomplete or {})
            raise KeyboardInterrupt          # stands in for a job killed mid-operation
    except KeyboardInterrupt:
        pass
    check('ledger.incomplete names the operation WHILE it runs',
          seen.get('during', {}).get('label') == 'the one that never returns',
          str(seen.get('during')))


def test_heartbeat_survives_a_nested_watchdog():
    led = diag.OpLedger('t', echo=False)
    path = os.path.join(tempfile.mkdtemp(), 'hb.log')
    open(path, 'w').close()
    with diag.heartbeat(1.0, path):
        with diag.timed('short op', ledger=led, timeout=30):
            time.sleep(0.2)
        time.sleep(2.6)
    n = sum(1 for ln in open(path) if ln.startswith('Timeout ('))
    check('the heartbeat keeps firing after a nested timed block exits', n >= 2,
          f'{n} dumps in 2.6 s at a 1 s interval')


def test_cuda_bookkeeping_is_keyed_on_the_target_device():
    check('a cpu device is not treated as cuda', not diag._is_cuda(torch.device('cpu')))
    check('None is not treated as cuda', not diag._is_cuda(None))
    if torch.cuda.is_available():
        check('a cuda device IS treated as cuda', diag._is_cuda(torch.device('cuda')))
    else:
        check('a cuda device is refused when no card is present',
              not diag._is_cuda(torch.device('cuda')))

    # The overhead check, which is what caught the bug: instrumentation must be negligible.
    led = diag.OpLedger('t', echo=False)
    t0 = time.time()
    for i in range(20):
        with diag.timed(f'op {i}', ledger=led, device=torch.device('cpu')):
            pass
    per_op_ms = (time.time() - t0) / 20 * 1e3
    check('per-operation overhead stays under 20 ms on a cpu device', per_op_ms < 20,
          f'{per_op_ms:.2f} ms/op')


if __name__ == '__main__':
    print('=' * 78 + '\ndiagnosis tooling\n' + '=' * 78)
    test_watchdog_fires_and_names_the_call()
    test_incomplete_names_the_running_op()
    test_heartbeat_survives_a_nested_watchdog()
    test_cuda_bookkeeping_is_keyed_on_the_target_device()
    print('\n' + '=' * 78)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1 if FAIL else 0)
