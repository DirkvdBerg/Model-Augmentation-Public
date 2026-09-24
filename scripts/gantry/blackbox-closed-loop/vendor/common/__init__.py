"""VENDOR-PATCH (BB-002): vendor-only file, no source counterpart.

The source `scripts/gantry/common/` is a NAMESPACE package (no __init__). The grey-box entry file
puts `scripts/gantry` first on sys.path, and a namespace package merges portions from every path
entry, so `common.oracle` resolved to the repo copy (run g0_infra, attempt 2). This file makes
the vendored `common` a regular package, pinned to this folder.
"""
