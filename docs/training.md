# Training and reproduction

The optimizations are environment switches on the upstream trainer. Every switch is off unless set to `1`; with none
set, the trainer runs the upstream code path. The measured configuration is
[`scripts/configs/faster.env`](../scripts/configs/faster.env); [`scripts/configs/faster_robotwin.env`](../scripts/configs/faster_robotwin.env)
is the same without the text-embedding memo.

## Environment

The upstream environment (Python 3.10, PyTorch 2.7.1 + CUDA 12.8, `pip install -e .`), see
[README.upstream.md](../README.upstream.md#environment-setup). The measurements here used torch 2.7.1+cu128,
torchvision 0.22.1+cu128, accelerate 1.12.0, DeepSpeed 0.18.5, transformers 4.49.0, PyAV 16.0.1 and NCCL 2.27.3
(driver 580.126.09).

**NCCL version.** PyTorch 2.7.1 pins `nvidia-nccl-cu12==2.26.2`. On the machines used here, every NCCL
communicator initialization with 2.26.2 hung (2 or 8 GPUs, with or without InfiniBand, P2P or NVLS); 2.27.3 (PyTorch
2.8's pin) worked. PyTorch loads `libnccl.so.2` at run time, so the library can be replaced in place:

```bash
pip install --no-deps nvidia-nccl-cu12==2.27.3
python -c "import ctypes, glob, os, torch
lib = ctypes.CDLL(glob.glob(os.path.join(os.path.dirname(torch.__file__), '..', 'nvidia', 'nccl', 'lib', 'libnccl.so.2'))[0])
v = ctypes.c_int(); lib.ncclGetVersion(ctypes.byref(v)); print(v.value)"   # 22703
```

`torch.cuda.nccl.version()` reports the version PyTorch was built with, not the library it loaded.

## Data and weights

As upstream: [model preparation](../README.upstream.md#model-preparation), [dataset download](../README.upstream.md#dataset-download)
and the [T5 embedding cache](../README.upstream.md#1-precompute-t5-embedding-cache-before-training). The cache holds one
file of about 1 MB per distinct instruction: 40 files for LIBERO, 921,032 files (about 0.97 TB) for RoboTwin 2.0.

## Launch

Source the switches, then use the upstream command. LIBERO trains on one node with 8 GPUs:

```bash
source scripts/configs/faster.env
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
```

Upstream trains RoboTwin on 64 GPUs with 16 samples per GPU, a global batch of 1,024. On one 8-GPU node, keep the
global batch with gradient accumulation:

```bash
source scripts/configs/faster_robotwin.env
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4 gradient_accumulation_steps=8
```

With the compile switches, 32 samples per GPU with `gradient_accumulation_steps=4` also fits in memory (about 106 GB
per GPU); 64 and 128 per GPU run out of memory in the first forward, and upstream's MoT does not support gradient
checkpointing (`model.mot_checkpoint_mixed_attn=true` raises). Without the compile switches, 32 per GPU runs out of
memory as well.

When `FASTWAM_OPT_COMPILE_DENOISE=1` is active, the trainer logs `Compiling training denoise forward/backward before
DeepSpeed initialization`; the VAE encode compiles during the first steps. Check both before trusting a timing.

## Switches

| Switch | What it does | Numerics |
|---|---|---|
| `FASTWAM_OPT_PRESAMPLE` | Decodes and transforms only the frames the model uses (9 of 33 per camera) instead of all of them; upstream already does this for LeRobot v3 data | Bit-identical |
| `FASTWAM_OPT_FASTROWS` | Rows and timestamps come from whole-column tensors built once at start, with the same conversion, instead of a Hugging Face `select` per item | Bit-identical |
| `FASTWAM_OPT_U8FRAMES` | The video decoder returns its uint8 frames; the dataset's float → uint8 round trip is the identity on all 256 values | Bit-identical |
| `FASTWAM_OPT_TEXTMEMO` | Each data-loader worker keeps the T5 cache entries it has loaded. Unbounded (about 1 MB per distinct instruction): for LIBERO, not for RoboTwin | Bit-identical |
| `FASTWAM_OPT_NOSYNC` | The loss dictionary stays on the GPU; metrics are gathered only on logging steps; batches are copied to the GPU without blocking | Bit-identical |
| `FASTWAM_OPT_VAE_INDUCTOR` | The VAE encode is compiled with Inductor instead of the CUDA-graph backend | Rounding differs |
| `FASTWAM_OPT_COMPILE_DENOISE` | Upstream's regional compile of the MoT layers; the same as `model.compile_training_denoise=true` | Rounding differs |

"Bit-identical" means that, with the switch set and the compile switches unset, all recorded tensors of a two-step
training run in deterministic mode (losses, outputs, gradients, parameter updates and AdamW state, 105,728 in all)
matched the baseline exactly. "Rounding differs": the compiled kernels sum in a different order; the final
configuration passed the statistical and training-run comparisons in the README. The compiled backward is not
bit-reproducible across processes either, even with deterministic algorithms.

Under DeepSpeed, `accelerator.clip_grad_norm_` only reads the gradient norm (clipping follows the DeepSpeed config,
which upstream leaves off), so `FASTWAM_OPT_NOSYNC` skips it on steps that do not log. With any other backend it
clips, and it runs on every step.

## Memory

LIBERO recipe, 16 samples per GPU: peak allocated memory per GPU **62.6 GiB (baseline) → 53.9 GiB (optimized)**.
RoboTwin at 16 samples per GPU: about 107 GB (baseline) per GPU.

## Benchmark protocol

- Workload: the upstream LIBERO recipe unchanged (`task=libero_uncond_2cam224_1e-4`, 8 GPUs × 16 samples, DeepSpeed
  ZeRO-1, BF16 mixed precision, AdamW at 1e-4 with cosine decay, no gradient clipping), data read from local disk.
- The timer runs the repository's real `scripts/train.py`, skips 32 warm-up iterations, then times two windows of
  64 iterations. Baseline → candidate → baseline in independent processes; each window and the random-number stream
  are checked against a frozen baseline run, and a deviating window invalidates the run.
- The speed bar was 1.10×, set from the baseline's own run-to-run spread on this machine (up to 7%).
- Validation, set before testing optimized configurations:
  - structure checks: the training entry point, the resolved configuration and digests of the data loader's batches
    are identical to the baseline;
  - exact tensor comparisons of a two-step deterministic run;
  - statistical comparisons of 160-step training runs with matched seeds 42-44 (up to 42-47): video and action losses
    over the last 50 steps, fixed-noise probe losses and parameter-update RMS over the last 10 steps, each against a
    threshold of twice the baseline standard deviation across six seeds.
- The checks were tested on deliberately wrong versions before use. They rejected changed weight decay, changed data
  normalization, skipped action noise, a detached action loss, a 1.2× action-loss weight and a 1.02× learning rate,
  and accepted an arithmetic reordering that is exact in real numbers.

The final configuration did not match the baseline bit for bit; the statistical comparison concluded "equivalent" after
three seeds. These measurements concern training speed and short training trajectories; the end-to-end LIBERO
evaluation in the README is a single run.

## Training-run comparison

The before/after runs in the README use the upstream recipes with FastWAM's official per-GPU batch of 16 and no
gradient accumulation: `task=robotwin_uncond_3cam_384_1e-4` for 40 steps and `task=libero_uncond_2cam224_1e-4` for 100
steps, `log_every=1`, 8 GPUs, weights compared at the last step. `run_training` builds the model before `Wan22Trainer`
seeds the random generators, so the randomly initialized modules (action head, action encoder, proprioception encoder)
start differently in every run. For the comparison, every tree compared, the upstream one included, called
`set_global_seed(cfg.seed)` just before `instantiate(cfg.model, ...)` in `run_training`. That line is a test harness,
not part of this repository's code. With it, two upstream runs matched bit for bit.

The code of this repository was checked the same way on LIBERO: with no switch set, it matched upstream bit for bit
(all 104 logged losses and all 1,651 weight tensors); with `scripts/configs/faster.env`, it differed from upstream as
much as the checked configuration did.
