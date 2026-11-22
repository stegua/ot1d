# Checks OT1D on Windows with the Microsoft compiler (MSVC): the C++ tests with
# several sets of compiler flags, the Python module built from source in a
# temporary virtual environment, and a short benchmark. It is the Windows
# version of tests/check_platform.sh.
#
# Run it from a "Developer PowerShell for VS" (Start menu), so that the compiler
# cl.exe is on the PATH, in an environment where "python" is Python 3.8 or
# later (for instance after "conda activate base"):
#
#   git checkout develop
#   git pull
#   powershell -ExecutionPolicy Bypass -File tests\check_platform.ps1
#   powershell -ExecutionPolicy Bypass -File tests\check_platform.ps1 -Quick
#
# -Quick skips the sanitizer build and the benchmark. Nothing is installed
# outside a temporary folder, which is removed at the end. Everything printed
# is also written to ot1d_platform_report.txt (in the root of the repository):
# that file is all that is needed to diagnose a failure.
# It needs a network connection for pip.

param([switch]$Quick)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Report = Join-Path $Root "ot1d_platform_report.txt"
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ot1d_check_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $Tmp | Out-Null
$script:Passed = @()
$script:Failed = @()

Start-Transcript -Path $Report -Force | Out-Null

function Section($name) {
    Write-Host ""
    Write-Host "==================== $name"
}

# Runs a script block, and records whether it passed: it fails if it throws, or
# if the last program it ran returned a nonzero exit code.
function Step($name, [scriptblock]$body) {
    Section $name
    $global:LASTEXITCODE = 0
    $ok = $true
    try {
        & $body
        if ($LASTEXITCODE -ne 0) { $ok = $false }
    } catch {
        Write-Host $_
        $ok = $false
    }
    if ($ok) {
        $script:Passed += "  ok      $name"
    } else {
        $script:Failed += "  FAILED  $name"
        Write-Host "*** FAILED: $name"
    }
}

# Compiles tests\test_ot1d.cpp with cl and the given flags, and runs it
function CxxTest($out, $flags) {
    $exe = Join-Path $Tmp "$out.exe"
    $obj = Join-Path $Tmp "$out.obj"
    $cmd = @("/nologo", "/EHsc", "/W3", "/std:c++14", "/Iinclude") + $flags + @("tests\test_ot1d.cpp", "/Fe:$exe", "/Fo:$obj")
    Write-Host ("cl " + ($cmd -join " "))
    & cl @cmd 2>&1 | Select-Object -Last 40 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "compilation failed" }
    & $exe 2>&1 | Select-Object -Last 15 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { throw "the tests failed" }
}

try {
    Section "Machine"
    Get-Date
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
        Write-Host ("{0} {1}" -f $os.Caption, $os.Version)
        Write-Host $cpu.Name
        Write-Host ("cores: {0}, memory: {1} GB" -f $env:NUMBER_OF_PROCESSORS, [math]::Round($os.TotalVisibleMemorySize / 1MB))
    } catch {
        Write-Host "(no system information: $_)"
    }
    Write-Host ("PowerShell " + $PSVersionTable.PSVersion.ToString())
    git log --format="commit %h %s" -1
    git status --short | Where-Object { $_ -notmatch '^\?\?' }

    $haveCl = $null -ne (Get-Command cl -ErrorAction SilentlyContinue)
    $havePython = $null -ne (Get-Command python -ErrorAction SilentlyContinue)
    if ($haveCl) {
        # cl prints its version banner on the standard error
        cmd /c "cl 2>&1" | Select-Object -First 2 | ForEach-Object { Write-Host $_ }
    } else {
        Write-Host "cl.exe not found: run this script from a 'Developer PowerShell for VS'."
    }
    if ($havePython) { python --version } else { Write-Host "python not found on the PATH." }

    Step "The two copies of the header are the same" {
        $a = Get-FileHash include\OT1D.hpp
        $b = Get-FileHash python\OT1D.hpp
        if ($a.Hash -ne $b.Hash) { throw "include\OT1D.hpp and python\OT1D.hpp differ" }
    }

    if ($haveCl) {
        # /O2 with the default /fp:precise are the flags of the Python module
        Step "C++ tests, /O2" { CxxTest "t_o2" @("/O2", "/DNDEBUG") }
        Step "C++ tests, /O2 /fp:fast" { CxxTest "t_fast" @("/O2", "/fp:fast", "/DNDEBUG") }
        Step "C++ tests, /O2 /arch:AVX2" { CxxTest "t_avx2" @("/O2", "/arch:AVX2", "/DNDEBUG") }
        if (-not $Quick) {
            Step "C++ tests, AddressSanitizer" { CxxTest "t_asan" @("/O1", "/Zi", "/fsanitize=address") }
        }
    } else {
        $script:Failed += "  FAILED  C++ tests (cl.exe not found)"
    }

    Section "Python: virtual environment and build from source"
    $py = $null
    if ($havePython) {
        python -m venv (Join-Path $Tmp "venv")
        $py = Join-Path $Tmp "venv\Scripts\python.exe"
        if (-not (Test-Path $py)) { $py = Join-Path $Tmp "venv/bin/python" }   # not Windows
    }
    $built = $false
    if ($py -and (Test-Path $py)) {
        & $py -m pip install -q --upgrade pip
        & $py -m pip install -q numpy scipy pytest
        if ($LASTEXITCODE -eq 0) {
            & $py -m pip install (Join-Path $Root "python") 2>&1 | Select-Object -Last 25 | ForEach-Object { Write-Host $_ }
            if ($LASTEXITCODE -eq 0) { $built = $true }
        }
    }
    if ($built) {
        $script:Passed += "  ok      Python: build from source"
        $testWrapper = Join-Path $Root "python\test_wrapper.py"
        $testReference = Join-Path $Root "python\test_reference.py"
        $bench = Join-Path $Root "tests\bench_platform.py"
        # Not in python\, so that the installed module is the one imported
        Set-Location $Tmp
        Step "Python: module information" {
            & $py -c "import ot1d, numpy, sys, platform; print('ot1d', ot1d.__version__, '| SIMD sorting:', ot1d.simd(), '| numpy', numpy.__version__, '| python', sys.version.split()[0], '|', platform.machine()); print('module:', ot1d.__file__)"
        }
        Step "Python: test_wrapper.py" { & $py $testWrapper }
        Step "Python: test_reference.py on the C++ module" {
            $env:OT1D_REFERENCE = "native"
            & $py -m pytest -q $testReference
            $code = $LASTEXITCODE
            Remove-Item Env:\OT1D_REFERENCE
            $global:LASTEXITCODE = $code
        }
        Step "Python: test_reference.py on the NumPy prototype" {
            $env:PYTHONPATH = Join-Path $Root "python"
            & $py -m pytest -q $testReference
            $code = $LASTEXITCODE
            Remove-Item Env:\PYTHONPATH
            $global:LASTEXITCODE = $code
        }
        if (-not $Quick) {
            Step "Benchmark" { & $py $bench }
        }
        Set-Location $Root
    } else {
        $script:Failed += "  FAILED  Python: build from source"
        Write-Host "*** FAILED: could not build the Python module"
    }

    Section "Summary"
    $script:Passed | ForEach-Object { Write-Host $_ }
    if ($script:Failed.Count -gt 0) {
        Write-Host ""
        $script:Failed | ForEach-Object { Write-Host $_ }
        Write-Host ""
        Write-Host "Some steps FAILED. Report: $Report"
    } else {
        Write-Host ""
        Write-Host "All steps passed. Report: $Report"
    }
} finally {
    Set-Location $Root
    Stop-Transcript | Out-Null
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
if ($script:Failed.Count -gt 0) { exit 1 }
exit 0
