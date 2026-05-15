# vgmgpt.ps1 — VgmGPT helper script
# Usage:
#   .\vgmgpt.ps1 train               — Resume (or start) v3 training
#   .\vgmgpt.ps1 gen ghz [STEP]      — Generate 10 GHZ tracks from checkpoint
#   .\vgmgpt.ps1 gen metal [STEP]    — Generate 10 Metal Squad tracks from checkpoint
#   .\vgmgpt.ps1 gen both [STEP]     — Generate both batches in parallel
#   .\vgmgpt.ps1 status              — Show current training step and latest checkpoints

param(
    [Parameter(Position=0)] [string]$Command,
    [Parameter(Position=1)] [string]$Sub,
    [Parameter(Position=2)] [string]$Step
)

Set-Location $PSScriptRoot

# ── helpers ──────────────────────────────────────────────────────────────────

function Get-LatestCheckpoint {
    $checkpoints = Get-ChildItem runs\v3\step_*.pt -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "latest.pt" } |
        Sort-Object Name
    if (-not $checkpoints) { Write-Error "No step_*.pt checkpoints found in runs\v3\"; exit 1 }
    return ($checkpoints | Select-Object -Last 1).Name -replace '\.pt$',''
}

function Resolve-Checkpoint([string]$StepArg) {
    if ($StepArg) {
        # Accept bare number (e.g. "36000") or full name (e.g. "step_036000")
        if ($StepArg -match '^\d+$') {
            return "runs/v3/step_{0:D6}.pt" -f [int]$StepArg
        } else {
            return "runs/v3/$StepArg.pt"
        }
    }
    $name = Get-LatestCheckpoint
    return "runs/v3/$name.pt"
}

# ── commands ─────────────────────────────────────────────────────────────────

switch ($Command) {

    "train" {
        Write-Host "Starting v3 training (auto-resumes from runs\v3\latest.pt if present)..."
        python -m genesis_music.train `
            --output-dir runs/v3 `
            --data-dir   data/prepared `
            --model-size large `
            --seq-len    8192 `
            --batch-size 4 `
            --grad-accum 8 `
            --max-steps  50000 `
            --no-compile `
            --gradient-checkpointing `
            2>&1 | Tee-Object -FilePath runs/v3/train.log -Append
    }

    "gen" {
        $ckpt = Resolve-Checkpoint $Step
        # Derive a label from the checkpoint filename for output naming
        $label = (Split-Path $ckpt -Leaf) -replace '\.pt$',''

        if (-not (Test-Path $ckpt)) {
            Write-Error "Checkpoint not found: $ckpt"; exit 1
        }

        New-Item -ItemType Directory -Path output/v3_progress -Force | Out-Null

        switch ($Sub) {

            "ghz" {
                Write-Host "Generating 10 GHZ tracks from $ckpt ..."
                python -m genesis_music.generate `
                    --checkpoint  $ckpt `
                    --prompt-vgm  "data/vgm/Sonic_the_Hedgehog__Mega_Drive__Genesis___02_-_Green_Hill_Zone.vgz" `
                    --prompt-tokens 512 `
                    --repetition-penalty 1.2 `
                    --repetition-window 128 `
                    --device cpu `
                    --n 10 `
                    --output "output/v3_progress/${label}_ghz_001.vgm"
            }

            "metal" {
                Write-Host "Generating 10 Metal Squad tracks from $ckpt ..."
                python -m genesis_music.generate `
                    --checkpoint  $ckpt `
                    --prompt-vgm  "data/vgm/Thunder_Force_IV__Lightening_Force___Mega_Drive__Genesis___23_-_Metal_Squad__Stage_8_.vgz" `
                    --prompt-tokens 4096 `
                    --repetition-penalty 1.2 `
                    --repetition-window 64 `
                    --device cpu `
                    --n 10 `
                    --output "output/v3_progress/${label}_metal_squad_p4096_001.vgm"
            }

            "both" {
                Write-Host "Generating GHZ + Metal Squad in parallel from $ckpt ..."

                $ghzJob = Start-Job -ScriptBlock {
                    param($ckpt, $label)
                    Set-Location D:\dev\genesis-music-ml
                    python -m genesis_music.generate `
                        --checkpoint  $ckpt `
                        --prompt-vgm  "data/vgm/Sonic_the_Hedgehog__Mega_Drive__Genesis___02_-_Green_Hill_Zone.vgz" `
                        --prompt-tokens 512 `
                        --repetition-penalty 1.2 `
                        --repetition-window 128 `
                        --device cpu `
                        --n 10 `
                        --output "output/v3_progress/${label}_ghz_001.vgm" 2>&1
                } -ArgumentList $ckpt, $label

                $metalJob = Start-Job -ScriptBlock {
                    param($ckpt, $label)
                    Set-Location D:\dev\genesis-music-ml
                    python -m genesis_music.generate `
                        --checkpoint  $ckpt `
                        --prompt-vgm  "data/vgm/Thunder_Force_IV__Lightening_Force___Mega_Drive__Genesis___23_-_Metal_Squad__Stage_8_.vgz" `
                        --prompt-tokens 4096 `
                        --repetition-penalty 1.2 `
                        --repetition-window 64 `
                        --device cpu `
                        --n 10 `
                        --output "output/v3_progress/${label}_metal_squad_p4096_001.vgm" 2>&1
                } -ArgumentList $ckpt, $label

                Write-Host "  GHZ job ID:   $($ghzJob.Id)"
                Write-Host "  Metal job ID: $($metalJob.Id)"
                Write-Host "Waiting for both to complete (this will take ~90 min on CPU)..."

                $results = $ghzJob, $metalJob | Wait-Job | Receive-Job
                $results | Write-Host
            }

            default {
                Write-Host "Usage: .\vgmgpt.ps1 gen <ghz|metal|both> [step]"
            }
        }
    }

    "status" {
        Write-Host "=== Latest training log ==="
        Get-Content runs\v3\train.log -Tail 5

        Write-Host ""
        Write-Host "=== Recent val_loss ==="
        Select-String "val_loss" runs\v3\train.log | Select-Object -Last 5 | ForEach-Object { $_.Line }

        Write-Host ""
        Write-Host "=== Checkpoints ==="
        Get-ChildItem runs\v3\step_*.pt | Sort-Object Name | Select-Object Name, LastWriteTime | Format-Table -AutoSize
    }

    default {
        Write-Host @"
vgmgpt.ps1 — VgmGPT helper

Commands:
  train               Resume (or start) v3 training
  gen ghz   [step]    Generate 10 GHZ tracks
  gen metal [step]    Generate 10 Metal Squad tracks
  gen both  [step]    Generate both batches in parallel
  status              Show training progress and checkpoints

[step] is optional. If omitted, uses the latest checkpoint automatically.
Examples:
  .\vgmgpt.ps1 train
  .\vgmgpt.ps1 gen both
  .\vgmgpt.ps1 gen ghz 36000
  .\vgmgpt.ps1 status
"@
    }
}
