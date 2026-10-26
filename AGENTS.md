# Repository instructions

- Keep the upstream workload: the task configs in `configs/task/` unchanged, 16 samples per GPU, DeepSpeed ZeRO-1. The 3.46× headline is an 8-GPU per-iteration benchmark of the LIBERO recipe; the only end-to-end timing is the 3 h 05 min LIBERO run. Do not extend either to other recipes or hardware without measuring.
- Keep optimization switches off by default. The five data and synchronization switches matched the baseline bit for bit in the recorded tests; the two compile switches change rounding and passed statistical and training-run comparisons. Do not claim lossless acceleration for them.
- `FASTWAM_OPT_TEXTMEMO` keeps every loaded T5 cache entry (~1 MB each) in every data-loader worker. Leave it unset for datasets with many distinct instructions, such as RoboTwin 2.0 (921,032).
- Full RoboTwin training and evaluation have not been run with these switches. Do not fill them in with paper numbers. Put new experiments in a new dated directory with source, dependency and hardware provenance.
- Check that the switches took effect before trusting a timing: with `FASTWAM_OPT_COMPILE_DENOISE=1` the trainer logs the training-denoise compile warmup before DeepSpeed initialization, and the first steps compile the VAE encode when `FASTWAM_OPT_VAE_INDUCTOR=1`.
- Check the runtime NCCL version (see `docs/training.md`): 2.26.2 hung at communicator initialization on the machines used in this project, 2.27.3 worked.
