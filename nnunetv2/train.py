from nnunetv2.run.run_training import run_training_entry
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=1
if __name__ == '__main__':
    run_training_entry()