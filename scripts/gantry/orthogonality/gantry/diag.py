"""Per-operation timing, memory and a stall watchdog.

WHY THIS EXISTS. `stage_V3` never returned on the development box, four times, and the only
thing that could be said about it afterwards was "V3 hangs". The one measurement that did carry
information was CPU time against wall clock: the process accumulated 16 to 40 s of CPU per
20 minutes of wall clock, about one per cent, which says BLOCKED rather than SLOW and rules out
"the Jacobian is expensive" without ruling anything in. Halving the problem, removing a dense
`N x N` projector, removing a `4800 x 6774` SVD and pinning threads all failed to move it, and the
stall point moved between attempts.

So this module records, per operation:

    wall clock          how long the caller waited
    CPU time            `time.process_time`. Wall clock without this cannot distinguish a slow
                        operation from a blocked one, which is the distinction that mattered
    peak RSS            `resource.getrusage`, which exists on Linux and not on Windows. This is
                        why the saved local results have `peak_memory_mb = nan`: it is a platform
                        artifact, not a missing measurement, and the server gets it for free
    CUDA peak           `torch.cuda.max_memory_allocated`, reset per operation so the figure is
                        the operation's own peak and not the run's

and, when an operation exceeds `timeout`, dumps the stack of every thread with
`faulthandler.dump_traceback_later`. That is the piece that turns "V3 hangs" into "V3 sits in
<named call>". The dump repeats, so a genuinely slow operation prints progressively rather than
once, and the caller can then abort the stage with a recorded failure instead of the job sitting
until SLURM kills it.

NOT A PROFILER. It is deliberately coarse: one record per named operation, no sampling, no
instrumentation of anything it is not explicitly wrapped around. A profiler would tell us where
the time goes in an operation that RUNS; the problem here is an operation that does not.
"""
__project_origin__ = "added"

import contextlib
import faulthandler
import os
import sys
import time

try:
    import resource            # Linux/macOS only; absent on Windows
except ImportError:            # pragma: no cover - platform dependent
    resource = None

# THE ACTIVE HEARTBEAT, if any: (interval, file object).
#
# `faulthandler` has exactly ONE global timer. `dump_traceback_later` replaces whatever was
# armed, and `cancel_dump_traceback_later` cancels it outright, so a per-operation watchdog
# nested inside a whole-run heartbeat SILENTLY DISABLES the heartbeat the first time an
# operation finishes. Measured: a 30-minute heartbeat produced an empty dump file across a
# 40-minute run because every `timed` block cancelled it on exit. Tracking it here lets `timed`
# put it back.
_HEARTBEAT = None


def peak_rss_mb():
    """Peak resident set size in MB, or nan where the platform cannot report it.

    `resource.getrusage` is the reliable route on Linux, which is where the server runs.
    `psutil` is tried first because it also works on Windows, but it is NOT in
    `environment.server-train.yml` today, so the fallback is what will actually fire.
    """
    try:
        import psutil
        return float(psutil.Process().memory_info().peak_wset) / 1e6
    except Exception:
        pass
    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        return float(ru) / (1e3 if sys.platform != 'darwin' else 1e6)
    return float('nan')


def _is_cuda(device):
    """True only when the TARGET device is CUDA.

    BUG FIXED 2026-09-08, found by this module's own per-operation ledger. The first version
    keyed on `torch.cuda.is_available()`, which is True on any machine with a card EVEN WHEN THE
    RUN IS ON THE CPU. Every timed operation then called `reset_peak_memory_stats` and
    `synchronize` against a CPU device, which forces CUDA context initialisation and a device
    synchronise for no reason. The symptom was diagnostic in itself: each operation reported
    ~105% CPU while the process as a whole accrued ~2%, i.e. the time was going BETWEEN the
    timed blocks, which is exactly where instrumentation overhead lives and exactly what a
    per-operation ledger is able to show.
    """
    if device is None:
        return False
    try:
        import torch
        return torch.device(device).type == 'cuda' and torch.cuda.is_available()
    except Exception:
        return False


def cuda_peak_mb(device=None):
    if not _is_cuda(device):
        return float('nan')
    try:
        import torch
        return float(torch.cuda.max_memory_allocated(device)) / 1e6
    except Exception:
        return float('nan')


def _cuda_reset(device=None):
    if not _is_cuda(device):
        return
    try:
        import torch
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    except Exception:
        pass


class OpLedger:
    """Ordered record of every timed operation, for the results JSON."""

    def __init__(self, name='', echo=True):
        self.name = name
        self.echo = echo
        self.rows = []
        self.incomplete = None            # the operation that never returned, if any

    def add(self, row):
        self.rows.append(row)
        if self.echo:
            print('  [op] {label:<38s} wall {wall:8.2f}s  cpu {cpu:8.2f}s  '
                  'cpu/wall {frac:5.1%}  rss {rss:8.1f}MB  cuda {cuda:8.1f}MB'.format(
                      label=row['label'][:38], wall=row['wall_s'], cpu=row['cpu_s'],
                      frac=row['cpu_fraction'], rss=row['peak_rss_mb'],
                      cuda=row['cuda_peak_mb']), flush=True)

    def summary(self):
        return dict(
            name=self.name, n_ops=len(self.rows), rows=self.rows,
            incomplete=self.incomplete,
            total_wall_s=sum(r['wall_s'] for r in self.rows),
            total_cpu_s=sum(r['cpu_s'] for r in self.rows),
            # The single most diagnostic number: an operation that blocks has a fraction near 0,
            # an operation that computes has a fraction near 1 (or above 1 with threads).
            min_cpu_fraction=min((r['cpu_fraction'] for r in self.rows), default=float('nan')),
        )


class OperationTimeout(RuntimeError):
    """Raised when a timed operation exceeds its budget, AFTER the stack has been dumped."""


@contextlib.contextmanager
def timed(label, ledger=None, timeout=None, device=None, dump_to=None, hard=False):
    """Time one named operation, and dump every thread's stack if it overruns.

    Parameters
    ----------
    timeout : seconds, or None to disable the watchdog.
    hard    : when True, the overrun raises `OperationTimeout` after the dump, so the caller
              records a failure and the ladder moves on. When False the dump repeats and the
              operation is left to finish, which is what you want the first time: the dump names
              the call, and the operation may still complete.

    The watchdog is `faulthandler.dump_traceback_later`, chosen because it fires from a C-level
    timer thread and therefore works even when the main thread is blocked inside a C call, which
    a Python-level timer would not.
    """
    _cuda_reset(device)
    fh = None
    if timeout:
        if dump_to:
            fh = open(dump_to, 'a', buffering=1)
            faulthandler.dump_traceback_later(timeout, repeat=not hard, exit=False, file=fh)
        else:
            faulthandler.dump_traceback_later(timeout, repeat=not hard, exit=False,
                                              file=sys.stderr)
    t0, c0 = time.time(), time.process_time()
    if ledger is not None:
        # Recorded BEFORE the body, so an operation that never returns still leaves a trace of
        # what was running. This is what was missing on every stalled attempt.
        ledger.incomplete = dict(label=label, started_at=time.strftime('%H:%M:%S'))
    try:
        yield
    finally:
        wall = time.time() - t0
        cpu = time.process_time() - c0
        if timeout:
            faulthandler.cancel_dump_traceback_later()
            _rearm_heartbeat()          # the timer is global; put back what we displaced
        if fh is not None:
            fh.close()
        row = dict(label=label, wall_s=wall, cpu_s=cpu,
                   cpu_fraction=(cpu / wall if wall > 0 else float('nan')),
                   peak_rss_mb=peak_rss_mb(), cuda_peak_mb=cuda_peak_mb(device),
                   timeout_s=timeout)
        if ledger is not None:
            ledger.incomplete = None
            ledger.add(row)
    if hard and timeout and row['wall_s'] > timeout:
        raise OperationTimeout(
            f'{label} took {row["wall_s"]:.1f}s against a {timeout}s budget, at '
            f'{row["cpu_fraction"]:.1%} CPU. A fraction near zero means BLOCKED, not slow; the '
            f'stack dump above names the call.')


def _rearm_heartbeat():
    """Re-arm the whole-run heartbeat after something else used the global timer."""
    if _HEARTBEAT is not None:
        interval, fh = _HEARTBEAT
        faulthandler.dump_traceback_later(interval, repeat=True, exit=False, file=fh)


@contextlib.contextmanager
def heartbeat(interval, path=None):
    """Dump every thread's stack every `interval` seconds, for the WHOLE run.

    The per-operation watchdog in `timed` covers the stage it is wrapped around. This covers
    everything else, and it exists because the stall did not stay put: across four attempts it
    appeared inside V3, inside V2, and once inside a plain no-grad rollout in V1. A watchdog on
    one stage would have missed two of those.

    Cheap by construction: `faulthandler` arms a C-level timer and prints only when it fires, so
    a run that never stalls pays one timer and produces no output. `repeat=True` means a genuine
    stall prints progressively, and the LAST dump before the job dies is the one that names the
    call.
    """
    global _HEARTBEAT
    if not interval:
        yield
        return
    fh = open(path, 'a', buffering=1) if path else sys.stderr
    _HEARTBEAT = (float(interval), fh)
    _rearm_heartbeat()
    try:
        yield
    finally:
        _HEARTBEAT = None
        faulthandler.cancel_dump_traceback_later()
        if path:
            fh.close()


def enable_fault_handler(path=None):
    """Install a SIGSEGV/SIGABRT handler and, on POSIX, a SIGUSR1 stack dump.

    SIGUSR1 lets a stalled job be interrogated from outside without killing it:
    `kill -USR1 <pid>` prints every thread's stack to the log. That is the cheapest possible
    remote diagnosis of a stall on the cluster, and it costs nothing when unused.

    NOTE: this does NOT pair with `#SBATCH --signal=USR1@...`, and the runners deliberately do not
    set that. Several scripts in this repo record why: without a handler installed, Python's
    default action for SIGUSR1 is to terminate, so the directive would kill the job. Here a
    handler IS installed, but the signal is meant to be sent by hand, not by SLURM at a deadline.
    """
    faulthandler.enable(file=open(path, 'a', buffering=1) if path else sys.stderr)
    if hasattr(faulthandler, 'register') and os.name == 'posix':
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
        return True
    return False
