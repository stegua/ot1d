# -*- coding: utf-8 -*-
"""Short benchmark of ot1d on this machine, run by tests/check_platform.sh (and .ps1).

  python tests/bench_platform.py                 # print the timings
  python tests/bench_platform.py --save          # also record them, and update the READMEs
  python tests/bench_platform.py --table         # only rebuild the table of the READMEs

With --save the timings are written to tests/platforms/<key>.json (the key is the name of the
CPU, or --key) and the table "Timings on different machines" of README.md and python/README.md
is rebuilt from all the files of that folder. The fields "machine" and "compiler" of an existing
file are kept (they can be edited by hand, or set with --name and --compiler).
"""
import argparse
import datetime
import glob
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORMS = os.path.join(ROOT, "tests", "platforms")
READMES = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "python", "README.md")]
BEGIN, END = "<!-- platforms: begin -->", "<!-- platforms: end -->"

# (key of the result, row of the table, unit)
ROWS = [
    ("np_sort", "`np.sort(x)`, 10^7 values", "s"),
    ("sort_1", "`ot1d.parasort(x)`, 10^7 values, 1 thread", "s"),
    ("sort_8", "`ot1d.parasort(x, threads=8)`", "s"),
    ("reference", "`ot1d.reference(x)`, 10^7 points", "s"),
    ("reference_weights", "`ot1d.reference(x, A)`, with weights", "s"),
    ("w1_ref_ecdf", "`w1_reference`, 10^4 against 10^7 points, `'ecdf'`", "ms"),
    ("w2_ref_ecdf", "`w2_reference`, same sizes, `'ecdf'`", "ms"),
    ("w1_ref_simplex", "`w1_reference`, same sizes, `'simplex'`", "ms"),
    ("w1_sorted_simplex", "`w1(x, y, sorting=False)`, 10^7 sorted points each", "ms"),
    ("w1_sorted_ecdf", "`w1(x, y, method='ecdf', sorting=False)`", "ms"),
    ("w1_unsorted", "`w1(x, y)`, 10^7 unsorted points each, 1 thread", "s"),
    ("batch_1", "`w1_batch`, per sample of 10^4 points, 1 thread", "ms"),
    ("batch_8", "`w1_batch`, per sample, 8 threads", "ms"),
]


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def cpu_name():
    s = platform.system()
    name = ""
    if s == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
        except OSError:
            pass
    elif s == "Darwin":
        name = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif s == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        except Exception:
            pass
    name = re.sub(r"\((R|TM|r|tm)\)", "", name or platform.processor() or platform.machine())
    return " ".join(name.split())


def memory_gb():
    s = platform.system()
    try:
        if s == "Windows":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in
                    ("total", "avail", "tpage", "apage", "tvirt", "avirt", "aext")]
            st = Status()
            st.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            total = st.total
        elif s == "Darwin":
            total = int(run(["sysctl", "-n", "hw.memsize"]))
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(round(total / 2.0**30))
    except Exception:
        return 0


def os_name():
    s = platform.system()
    if s == "Darwin":
        return "macOS " + platform.mac_ver()[0]
    if s == "Windows":
        return "Windows " + platform.release()
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return s


def load_all():
    out = []
    for p in sorted(glob.glob(os.path.join(PLATFORMS, "*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def cell(d, key):
    v = d.get("results", {}).get(key)
    if v is None:
        return "to be measured"
    return "{:.3g}".format(v) if v < 100 else "{:.0f}".format(v)


def table():
    ds = load_all()
    lines = [BEGIN,
             "| | " + " | ".join(d["machine"] for d in ds) + " |",
             "|:--|" + "--:|" * len(ds)]

    def line(label, f):
        lines.append("| " + label + " | " + " | ".join(f(d) for d in ds) + " |")

    line("CPU", lambda d: d["cpu"])
    line("Cores (logical), memory", lambda d: "{}, {} GB".format(d["cores"], d["memory_gb"]))
    line("System", lambda d: d["os"])
    line("Compiler", lambda d: d.get("compiler") or "")
    line("Python, NumPy", lambda d: ", ".join(v for v in (d.get("python"), d.get("numpy")) if v))
    line("SIMD sorting", lambda d: d.get("simd") or "to be measured")
    line("Version of ot1d, date", lambda d: "{}, {}".format(d["ot1d"], d["date"])
         if d.get("results") else "to be measured")
    for key, label, unit in ROWS:
        line("{} ({})".format(label, unit), lambda d, key=key: cell(d, key))
    lines.append(END)
    return "\n".join(lines)


def update_readmes():
    text = table()
    for p in READMES:
        with open(p, encoding="utf-8") as f:
            s = f.read()
        if BEGIN not in s or END not in s:
            sys.exit("markers of the table not found in " + p)
        s = s[:s.index(BEGIN)] + text + s[s.index(END) + len(END):]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(s)
        print("updated " + p)


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--save", action="store_true", help="record the timings and update the READMEs")
ap.add_argument("--table", action="store_true", help="only rebuild the table of the READMEs")
ap.add_argument("--key", help="name of the file in tests/platforms (default: from the CPU)")
ap.add_argument("--name", help="name of the machine in the table")
ap.add_argument("--compiler", help="compiler used to build ot1d")
args = ap.parse_args()
if args.table:
    update_readmes()
    sys.exit(0)

import numpy as np
import ot1d

RES = {}


def med(f, reps=7):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def row(name, seconds, unit="ms", key=None):
    scale = 1e3 if unit == "ms" else 1.0
    if key:
        RES[key] = float("{:.4g}".format(seconds * scale))
    print("  {:58s} {:10.3f} {}".format(name, seconds * scale, unit))


cores = os.cpu_count() or 1
r = np.random.default_rng(7)
n, m = 10_000_000, 10_000
x = r.lognormal(0, 1, n)
a = r.random(n) + 0.1
y = r.lognormal(0.1, 1, m)
b = r.random(m) + 0.1
xs, ys = np.sort(x), np.sort(r.lognormal(0.1, 1, n))

print("SIMD sorting: {}, cores: {}".format(ot1d.simd(), cores))
print("Reference of 10^7 points (README: sort 0.08 s, prepare 0.11 s, 0.06 s if sorted, 0.32 s with weights)")
row("np.sort", med(lambda: np.sort(x), 5), "s", key="np_sort")
row("ot1d.reference(x)", med(lambda: ot1d.reference(x), 5), "s", key="reference")
row("ot1d.reference(x, sorting=False), x sorted", med(lambda: ot1d.reference(xs, sorting=False), 5), "s")
row("ot1d.reference(x, A)", med(lambda: ot1d.reference(x, a), 5), "s", key="reference_weights")

R, RW = ot1d.reference(x), ot1d.reference(x, a)
print("One sample of 10^4 points against it (README: 0.076, 0.051, 0.20, 0.30, 10 ms)")
row("w1_reference, ecdf", med(lambda: ot1d.w1_reference(R, y), 51), key="w1_ref_ecdf")
row("w2_reference, ecdf", med(lambda: ot1d.w2_reference(R, y), 51), key="w2_ref_ecdf")
row("w1_reference, ecdf, weighted sample", med(lambda: ot1d.w1_reference(R, y, b), 51))
row("w1_reference, ecdf, weighted reference", med(lambda: ot1d.w1_reference(RW, y), 51))
row("w1_reference, simplex", med(lambda: ot1d.w1_reference(R, y, method="simplex"), 21), key="w1_ref_simplex")

print("Two samples of 10^7 points (README, sorted input: simplex 9 ms, OT1D 8.5 ms, ecdf 84 ms)")
row("w1, simplex, sorting=False", med(lambda: ot1d.w1(xs, ys, sorting=False), 9), key="w1_sorted_simplex")
row("OT1D, sorting=False", med(lambda: ot1d.OT1D(xs, ys, p=1, sorting=False), 9))
row("w1, ecdf, sorting=False", med(lambda: ot1d.w1(xs, ys, method="ecdf", sorting=False), 5), key="w1_sorted_ecdf")
row("w1, simplex, unsorted, threads=1", med(lambda: ot1d.w1(x, ys[::-1], threads=1), 3), "s", key="w1_unsorted")

print("Threads (parallel sort) on 10^7 points")
for t in sorted({1, 2, 4, 8, min(16, cores)}):
    if t > cores:
        continue
    t1 = med(lambda: ot1d.parasort(x.copy(), threads=t), 3) - med(lambda: x.copy(), 3)
    t2 = med(lambda: ot1d.w1_plan(x[:2_000_000], y, threads=t), 3)
    if t in (1, 8):
        RES["sort_{}".format(t)] = float("{:.4g}".format(t1))
    print("  threads={:2d}: sort of 10^7 values {:7.3f} s   w1_plan, 2*10^6 against 10^4 points {:7.3f} s".format(t, t1, t2))

chunks = [r.lognormal(0.1, 1, m) for _ in range(200)]
print("w1_batch, 200 samples of 10^4 points")
for t in sorted({1, min(8, cores)}):
    row("threads={}".format(t), med(lambda: ot1d.w1_batch(R, chunks, threads=t), 5) / len(chunks),
        key="batch_{}".format(t) if t in (1, 8) else None)

if args.save:
    cpu = cpu_name()
    key = args.key or re.sub(r"[^a-z0-9]+", "-", cpu.lower()).strip("-")
    path = os.path.join(PLATFORMS, key + ".json")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    d = {
        "machine": args.name or old.get("machine") or platform.node().split(".")[0],
        "cpu": cpu,
        "cores": cores,
        "memory_gb": memory_gb(),
        "os": os_name(),
        "compiler": args.compiler or old.get("compiler") or "",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ot1d": ot1d.__version__,
        "simd": ot1d.simd(),
        "date": datetime.date.today().isoformat(),
        "results": RES,
    }
    os.makedirs(PLATFORMS, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        json.dump(d, f, indent=2)
        f.write("\n")
    print("saved " + path)
    update_readmes()
