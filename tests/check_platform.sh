#!/usr/bin/env bash
#
# Checks OT1D on this machine (written for a first run on macOS with Apple
# Silicon, it also works on Linux): the C++ tests with several sets of compiler
# flags, the Python module built from source in a temporary virtual
# environment, and a short benchmark.
#
#   git checkout develop && git pull
#   bash tests/check_platform.sh            # about 5 to 10 minutes
#   bash tests/check_platform.sh --quick    # without sanitizers and benchmark
#
# Nothing is installed outside a temporary folder, which is removed at the end.
# Everything printed is also written to ot1d_platform_report.txt (in the root
# of the repository): that file is all that is needed to diagnose a failure.
# It needs a C++ compiler (on macOS: xcode-select --install), python3 (3.8 or
# later), and a network connection for pip.

set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
REPORT="$ROOT/ot1d_platform_report.txt"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1
CXX=${CXX:-c++}
PYTHON=${PYTHON:-python3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FAILED=""
PASSED=""

exec > >(tee "$REPORT") 2>&1

section() { printf '\n==================== %s\n' "$1"; }

# step NAME COMMAND...: runs the command, records whether it passed
step() {
    local name=$1
    shift
    section "$name"
    echo "\$ $*"
    if "$@"; then
        PASSED="$PASSED\n  ok      $name"
    else
        FAILED="$FAILED\n  FAILED  $name"
        echo "*** FAILED: $name"
    fi
}

cxx_test() {  # cxx_test OUTPUT FLAGS...
    local out=$1
    shift
    "$CXX" "$@" -Wall -std=c++11 -pthread tests/test_ot1d.cpp -o "$TMP/$out" -Iinclude && "$TMP/$out" | tail -n 12
    return "${PIPESTATUS[0]}"
}

section "Machine"
date
uname -a
if [ "$(uname)" = "Darwin" ]; then
    sysctl -n machdep.cpu.brand_string
    echo "cores: $(sysctl -n hw.ncpu), memory: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
    sw_vers
else
    grep -m1 "model name" /proc/cpuinfo
    echo "cores: $(nproc)"
fi
"$CXX" --version
"$PYTHON" --version
git log --format='commit %h %s' -1
git status --short | grep -v '^??' || true

step "The two copies of the header are the same" cmp include/OT1D.hpp python/OT1D.hpp

# The flags of the Python module: the compensated sums must survive -ffast-math
step "C++ tests, -O2 -ffast-math" cxx_test t_fast -O2 -ffast-math -DNDEBUG
step "C++ tests, -O2" cxx_test t_plain -O2 -DNDEBUG
step "C++ tests, -O3 -ffast-math, tuned for this CPU" cxx_test t_native -O3 -ffast-math -mcpu=native -DNDEBUG
if [ "$(uname -m)" = "x86_64" ]; then
    step "C++ tests with SIMD sorting (make test-simd)" make test-simd
fi
if [ $QUICK = 0 ]; then
    step "C++ tests, AddressSanitizer and UndefinedBehaviorSanitizer" \
        cxx_test t_asan -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
    step "C++ tests, ThreadSanitizer" cxx_test t_tsan -O1 -g -fsanitize=thread
fi

section "Python: virtual environment and build from source"
PY="$TMP/venv/bin/python"
if "$PYTHON" -m venv "$TMP/venv" && "$PY" -m pip install -q --upgrade pip &&
    "$PY" -m pip install -q numpy scipy pytest && "$PY" -m pip install "$ROOT/python" 2>&1 | tail -n 3; then
    PASSED="$PASSED\n  ok      Python: build from source"
    cd "$TMP"   # not in python/, so that the installed module is the one imported
    step "Python: module information" "$PY" -c "
import ot1d, numpy, sys, platform
print('ot1d', ot1d.__version__, '| SIMD sorting:', ot1d.simd(), '| numpy', numpy.__version__,
      '| python', sys.version.split()[0], '|', platform.machine())
print('module:', ot1d.__file__)"
    step "Python: test_wrapper.py" "$PY" "$ROOT/python/test_wrapper.py"
    step "Python: test_reference.py on the C++ module" env OT1D_REFERENCE=native "$PY" -m pytest -q "$ROOT/python/test_reference.py"
    step "Python: test_reference.py on the NumPy prototype" env PYTHONPATH="$ROOT/python" "$PY" -m pytest -q "$ROOT/python/test_reference.py"
    if [ $QUICK = 0 ]; then
        step "Benchmark" "$PY" "$ROOT/tests/bench_platform.py"
    fi
    cd "$ROOT"
else
    FAILED="$FAILED\n  FAILED  Python: build from source"
    echo "*** FAILED: could not build the Python module"
fi

section "Summary"
printf "%b\n" "$PASSED"
if [ -n "$FAILED" ]; then
    printf "%b\n" "$FAILED"
    echo
    echo "Some steps FAILED. Report: $REPORT"
    exit 1
fi
echo
echo "All steps passed. Report: $REPORT"
