param([ValidateSet('MECH_GATED','POSITIVE_CONTROL_LONG')][string]$JobMode = 'MECH_GATED')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

$root = 'C:\AI\nanochat_mech'
$main = 'C:\AI\nanochat'
$trainRoot = if ($JobMode -eq 'MECH_GATED') { $root } else { $main }
$python = Join-Path $main '.venv\Scripts\python.exe'
$statusPath = 'C:\AI\nanochat\STATUS_followup.md'
$lockPath = Join-Path $root 'run_followup.pid'
$logPath = Join-Path $root 'run_followup.log'
$maxResumes = 3
Set-Location -LiteralPath $trainRoot
$env:NANOCHAT_DATASET = 'fineweb-edu'; $env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'

function Write-Status([string]$sentence, [string]$job, [string]$progress, [string]$eta, [string]$errorText, [string]$decision) {
    $tmp = "$statusPath.tmp.$PID"
    $body = @($sentence,"job: $job","progress: $progress","ETA: $eta","last error: $errorText","decision inputs: $decision") -join "`r`n"
    [IO.File]::WriteAllText($tmp, $body + "`r`n", [Text.UTF8Encoding]::new($false)); Move-Item -LiteralPath $tmp -Destination $statusPath -Force
}
function Add-LogLines([string[]]$lines) {
    if ($lines.Count -eq 0) { return }
    $s=[IO.File]::Open($logPath,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::ReadWrite)
    try {$w=[IO.StreamWriter]::new($s,[Text.UTF8Encoding]::new($false));foreach($l in $lines){$w.WriteLine($l)};$w.Flush();$w.Dispose()}finally{$s.Dispose()}
}
function Get-LatestCompleteCheckpoint([string]$dir) {
    $cand=Get-ChildItem $dir -Filter 'model_*.pt' -File -ErrorAction SilentlyContinue | ForEach-Object {if($_.Name -match '^model_(\d+)\.pt$'){[int]$Matches[1]}} | Sort-Object -Descending
    foreach($step in $cand){$s='{0:D6}' -f $step;$need=@("model_$s.pt","meta_$s.json","optim_${s}_rank0.pt","rng_${s}_rank0.pt");if(@($need|Where-Object{-not(Test-Path(Join-Path $dir $_))}).Count){continue};try{if([int](Get-Content(Join-Path $dir "meta_$s.json") -Raw|ConvertFrom-Json).step -eq $step){return $step}}catch{}}
    return $null
}
function Assert-GpuChild {
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $probeOut = Join-Path $root 'gpu_probe.stdout.log'; $probeErr = Join-Path $root 'gpu_probe.stderr.log'
    $probe = 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
    Push-Location $trainRoot
    try { & $python -c $probe 1> $probeOut 2> $probeErr; $probeExit=$LASTEXITCODE } finally { Pop-Location }
    $device = if (Test-Path $probeOut) { ([IO.File]::ReadAllText($probeOut)).Trim() } else { '' }
    if ($probeExit -ne 0 -or [string]::IsNullOrWhiteSpace($device)) {
        $detail = if (Test-Path $probeErr) { (Get-Content $probeErr -Raw).Trim() } else { 'no probe output' }
        Add-LogLines @("STATUS GPU_PREFLIGHT_FAILED exit=$probeExit detail=$detail")
        Write-Status 'GPU preflight failed; no training child launched.' $JobMode 'failed' 'none' "GPU probe failed: $detail" 'fail closed'
        throw "GPU preflight failed: $detail"
    }
    Add-LogLines @("GPU_PREFLIGHT_OK device=$device")
}
function Run-Train([string]$label,[string]$jobDir,[string[]]$baseArgs,[int]$killAfterSeconds=0) {
    $resume=Get-LatestCompleteCheckpoint $jobDir;$exit=-1;$resumeCount=0
    for($attempt=1;$attempt -le ($maxResumes+1) -and $exit -ne 0;$attempt++){
        $args=@($baseArgs);if($null -ne $resume){$args+="--resume-from-step=$resume"};$out=Join-Path $jobDir "train_$attempt.stdout.log";$err=Join-Path $jobDir "train_$attempt.stderr.log"
        Add-LogLines @("TRAIN_START label=$label attempt=$attempt resume_step=$(if($null -eq $resume){'fresh'}else{$resume})");Write-Status "Follow-up training is running." $label "attempt $attempt" "unknown" "" "resume=$resume"
        Assert-GpuChild
        Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        $p=Start-Process $python -ArgumentList ($args -join ' ') -WorkingDirectory $trainRoot -RedirectStandardOutput $out -RedirectStandardError $err -PassThru;$t0=Get-Date;$lastLen=0;$lastGrowth=Get-Date;$lastCkpt=Get-Date
        while(-not $p.HasExited){Start-Sleep 5;$p.Refresh();if(Test-Path $out){$len=(Get-Item $out).Length;if($len -ne $lastLen){$lastLen=$len;$lastGrowth=Get-Date}};$latest=Get-LatestCompleteCheckpoint $jobDir;if($null -ne $latest){$lastCkpt=Get-Date};if($killAfterSeconds -gt 0 -and ((Get-Date)-$t0).TotalSeconds -ge $killAfterSeconds){Stop-Process -Id $p.Id -Force;Add-LogLines @("SMOKE_KILL pid=$($p.Id)");break};if(((Get-Date)-$lastGrowth).TotalMinutes -ge 30 -and ((Get-Date)-$lastCkpt).TotalMinutes -ge 45){Stop-Process -Id $p.Id -Force;Add-LogLines @("WATCHDOG_KILL pid=$($p.Id)");break}}
        $p.Refresh();$exit=$p.ExitCode;if(Test-Path $out){Add-LogLines @(Get-Content $out -Encoding UTF8)};if(Test-Path $err){Add-LogLines @(Get-Content $err -Encoding UTF8)};Add-LogLines @("TRAIN_END label=$label attempt=$attempt exit=$exit");if($exit -eq 0){break};$resume=Get-LatestCompleteCheckpoint $jobDir;if($null -eq $resume -or $resumeCount -ge $maxResumes){throw "Training $label failed exit=$exit checkpoint=$resume"};$resumeCount++
    }
    if($exit -ne 0){throw "Training $label failed exit=$exit"};return (Get-LatestCompleteCheckpoint $jobDir)
}
function Run-Smoke([string]$jobDir,[string[]]$baseArgs) {
    $resume=Get-LatestCompleteCheckpoint $jobDir
    if ($null -ne $resume) { throw "Smoke directory unexpectedly has checkpoint $resume" }
    $out=Join-Path $jobDir 'train_1.stdout.log'; $err=Join-Path $jobDir 'train_1.stderr.log'
    Assert-GpuChild
    Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    Add-LogLines @('TRAIN_START label=smoke attempt=1 resume_step=fresh')
    $p=Start-Process $python -ArgumentList ($baseArgs -join ' ') -WorkingDirectory $trainRoot -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $seen15=$false; $t0=Get-Date
    while(-not $p.HasExited) {
        Start-Sleep 5; $p.Refresh()
        if (Test-Path $out) {
            $lines=Get-Content $out -Encoding UTF8
            if (@($lines | Where-Object {$_ -match 'step\s+0*15/0*20'}).Count -gt 0) { $seen15=$true; if (Get-Process -Id $p.Id -ErrorAction SilentlyContinue) { Stop-Process -Id $p.Id -Force; Add-LogLines @("SMOKE_KILL_ITER15 pid=$($p.Id)") } else { Add-LogLines @("SMOKE_CHILD_ALREADY_EXITED pid=$($p.Id)") }; break }
        }
        if (((Get-Date)-$t0).TotalMinutes -ge 10) { Stop-Process -Id $p.Id -Force; throw 'Smoke gate timeout before iteration 15.' }
    }
    try { Wait-Process -Id $p.Id -Timeout 30 -ErrorAction SilentlyContinue } catch {}
    $p.Refresh(); if (Test-Path $out) { Add-LogLines @(Get-Content $out -Encoding UTF8) }; if (Test-Path $err) { Add-LogLines @(Get-Content $err -Encoding UTF8) }
    $resume=Get-LatestCompleteCheckpoint $jobDir
    if (-not $seen15 -or $null -eq $resume -or $resume -lt 10) { throw "Smoke gate failed: seen15=$seen15 checkpoint=$resume" }
    Add-LogLines @("SMOKE_RESTART checkpoint=$resume")
    $args=@(); $args += $baseArgs; $args += "--resume-from-step=$resume"; $out2=Join-Path $jobDir 'train_2.stdout.log'; $err2=Join-Path $jobDir 'train_2.stderr.log'
    Assert-GpuChild; Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    $p2=Start-Process $python -ArgumentList ($args -join ' ') -WorkingDirectory $trainRoot -RedirectStandardOutput $out2 -RedirectStandardError $err2 -PassThru -Wait
    if (Test-Path $out2) { Add-LogLines @(Get-Content $out2 -Encoding UTF8) }; if (Test-Path $err2) { Add-LogLines @(Get-Content $err2 -Encoding UTF8) }
    $all = if (Test-Path $out2) { Get-Content $out2 -Encoding UTF8 } else { @() }
    $finite = @($all | Where-Object {$_ -match 'step\s+0*20/0*20' -and $_ -notmatch 'nan|inf'}).Count -gt 0
    if ($p2.ExitCode -ne 0 -or -not $finite) { throw "Smoke restart failed exit=$($p2.ExitCode) finite_step20=$finite" }
    Add-LogLines @("SMOKE_PASS restart_exit=$($p2.ExitCode) iteration20_finite=$finite")
}
function Eval-Job([string]$label,[string]$jobDir,[int]$step,[int]$seed) {
    $wrap=Join-Path $jobDir 'eval_nomask_wrapper.py'
    @'
import sys
sys.path.insert(0, r"C:\AI\nanochat")
from diagnostics import recall_models
real = recall_models.load_model
def load_model(checkpoint_dir, step, device, seed, segment_len=128, decay_override=None):
    return real(checkpoint_dir, step, device, seed, segment_len=0, decay_override=decay_override)
import diagnostics.eval_recall as e
e.load_model = load_model
e.main()
'@ | Set-Content $wrap -Encoding UTF8
    $out=Join-Path $jobDir 'eval.stdout.log';$err=Join-Path $jobDir 'eval.stderr.log';$result=Join-Path $jobDir 'eval_result.txt';$args=@($wrap,'--checkpoint-dir',$jobDir,'--step',$step,'--device-batch-size','2','--n-rows','8192','--seed',$seed,'--eval-seed-base','3000000000','--allow-cross-above-0.09')
    $p=Start-Process $python -ArgumentList ($args -join ' ') -WorkingDirectory $trainRoot -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -Wait
    $text=[IO.File]::ReadAllText($out,[Text.Encoding]::UTF8)+"`r`n===== STDERR =====`r`n"+[IO.File]::ReadAllText($err,[Text.Encoding]::UTF8);[IO.File]::WriteAllText($result,$text,[Text.UTF8Encoding]::new($false));Add-LogLines @("EVAL_END label=$label exit=$($p.ExitCode) result=$result");if($p.ExitCode -ne 0){throw "Evaluation $label failed exit=$($p.ExitCode)"}
}

$oldPid = 0
if (Test-Path $lockPath) {
    try { $oldPid = [int](Get-Content $lockPath -ErrorAction Stop | Select-Object -First 1) } catch { $oldPid = 0 }
    if ($oldPid -gt 0 -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        throw "Another follow-up instance is live (PID $oldPid)."
    }
}
$lock=[IO.File]::Open($lockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
try {
    $lock.SetLength(0);$w=[IO.StreamWriter]::new($lock,[Text.UTF8Encoding]::new($false),1024,$true);$w.WriteLine($PID);$w.Flush();$w.Dispose();Add-LogLines @("RUN_START mode=$JobMode pid=$PID");Write-Status "Follow-up job has started." $JobMode "preparing" "unknown" "" "no mechanism test launched"
    if($JobMode -eq 'MECH_GATED'){$smoke=Join-Path $root 'followup-smoke-gated';New-Item -ItemType Directory -Force $smoke|Out-Null;$smokeArgs=@('-m','scripts.base_train','--model-tag',$smoke,'--seed','43','--depth=2','--aspect-ratio=32','--head-dim=32','--max-seq-len=128','--window-pattern=L','--device-batch-size=1','--total-batch-size=16384','--num-iterations=20','--save-every=10','--eval-every=-1','--core-metric-every=-1','--sample-every=-1','--rda','--rda-gated-write');Run-Smoke $smoke $smokeArgs}
    $names=if($JobMode -eq 'MECH_GATED'){@('followup-gated-M-s43-w20','followup-gated-M-const-s43-w20')}else{@('recall-A-nomask-long-s43-w20')}
    foreach($name in $names){$dir=Join-Path $trainRoot $name;New-Item -ItemType Directory -Force $dir|Out-Null;$iters=if($JobMode -eq 'MECH_GATED'){'1200'}else{'3600'};$args=@('-m','scripts.base_train','--model-tag',$dir,'--seed','43','--depth=8','--aspect-ratio=96','--head-dim=128','--max-seq-len=1024','--window-pattern=L','--device-batch-size=2','--total-batch-size=32768',"--num-iterations=$iters",'--eval-every=300','--eval-tokens=524288','--core-metric-every=-1','--sample-every=-1','--synthetic-recall-frac=0.5','--recall-answer-weight=20');if($JobMode -eq 'MECH_GATED'){$args+='--segment-len=128';$args+='--rda';$args+='--rda-gated-write';if($name -like '*const*'){$args+='--rda-const-state'}};Run-Train $name $dir $args|Out-Null;Eval-Job $name $dir ([int]$iters) 43}
    Write-Status "Follow-up job completed." $JobMode "complete" "none" "" "eval_result.txt files saved before logging"
}catch{$detail=($_ | Out-String).Trim();Add-LogLines @("RUN_FAIL $($_.Exception.Message)","RUN_FAIL_DETAIL $detail");Write-Status "Follow-up job stopped with an error." $JobMode "failed" "none" $_.Exception.Message "no further GPU job";throw}finally{$lock.Dispose()}
