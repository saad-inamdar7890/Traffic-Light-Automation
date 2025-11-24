<#
PowerShell Local Benchmark for SUMO + MAPPO
Usage:
  Open PowerShell in the repo root and run:
    .\s1\local_benchmark.ps1 -SUMOPath "C:\Path\to\sumo.exe" -Threads 4

If SUMO is on PATH, you can omit -SUMOPath.

This script will:
 - Print CPU and GPU info
 - Run a single SUMO simulation (using k1.sumocfg) with the specified threads and measure time
 - Run a single training episode (`--num-episodes 1`) and measure time
 - Output results to console and to `s1\benchmark_results.txt`
#>
param(
    [string]$SUMOPath = "sumo",
    [int]$Threads = 4,
    [string]$PythonExe = "python"
)

$outFile = Join-Path -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) -ChildPath "benchmark_results.txt"
"Local benchmark started at $(Get-Date -Format o)" | Tee-Object $outFile

Write-Host "\n=== System Info ===" -ForegroundColor Cyan
Get-CimInstance -ClassName Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | Format-List | Tee-Object -Append $outFile
Get-CimInstance Win32_VideoController | Select-Object Name,@{Name='AdapterRAMMB';Expression={[math]::Round($_.AdapterRAM/1MB,2)}} | Format-List | Tee-Object -Append $outFile

# Show Python + PyTorch info if available
Write-Host "\n=== Python & PyTorch Info ===" -ForegroundColor Cyan
try {
    $pyInfo = & $PythonExe -c "import platform,torch; print('Python', platform.python_version()); print('Torch', torch.__version__); print('CUDA', torch.cuda.is_available())"
    $pyInfo | Tee-Object -Append $outFile
    $pyInfo | Write-Host
} catch {
    "Python/PyTorch check failed: $_" | Tee-Object -Append $outFile
}

# SUMO benchmark
Write-Host "\n=== SUMO Benchmark ===" -ForegroundColor Cyan
$sumoCmd = "$SUMOPath -c .\s1\k1.sumocfg --no-warnings --no-step-log --threads $Threads"
"Running: $sumoCmd" | Tee-Object -Append $outFile

$sumoTime = Measure-Command { & $SUMOPath -c .\s1\k1.sumocfg --no-warnings --no-step-log --threads $Threads }
"SUMO elapsed: $($sumoTime.TotalSeconds) seconds" | Tee-Object -Append $outFile
Write-Host "SUMO elapsed: $($sumoTime.TotalSeconds) seconds"

# Training benchmark: 1 episode
Write-Host "\n=== Training Benchmark (1 episode) ===" -ForegroundColor Cyan
$trainCmd = "$PythonExe .\s1\mappo_k1_implementation.py --num-episodes 1"
"Running: $trainCmd" | Tee-Object -Append $outFile

$trainTime = Measure-Command { & $PythonExe .\s1\mappo_k1_implementation.py --num-episodes 1 }
"Training elapsed: $($trainTime.TotalSeconds) seconds" | Tee-Object -Append $outFile
Write-Host "Training elapsed: $($trainTime.TotalSeconds) seconds"

"\nBenchmark complete at $(Get-Date -Format o)" | Tee-Object -Append $outFile
Write-Host "Results saved to: $outFile" -ForegroundColor Green
