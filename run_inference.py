from config import CONFIG
import os

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

job_dir   = os.path.join(os.getcwd(), ".job")
out_dir   = os.path.join(os.getcwd(), ".out")
err_dir   = os.path.join(os.getcwd(), ".error")

mkdir_p(job_dir)
mkdir_p(out_dir)
mkdir_p(err_dir)

name = f"{CONFIG['run_name']}"
job_file = os.path.join(job_dir, f"{name}.job")

print(f"Writing SLURM job: {job_file}")
with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines(f"#SBATCH -J {name}\n")
    fh.writelines(f"#SBATCH -o .out/{name}.%j.out\n")
    fh.writelines(f"#SBATCH -e .error/{name}.%j.err\n")
    fh.writelines("#SBATCH -N 1\n")
    fh.writelines("#SBATCH --ntasks=1\n")
    fh.writelines("#SBATCH --cpus-per-task=16\n")
    fh.writelines("#SBATCH -p gpu-a100\n")
    fh.writelines("#SBATCH -A DPP20001\n")
    fh.writelines("#SBATCH -t 10:00:00\n\n")


    fh.writelines("echo \"Starting job on $(hostname) at $(date)\"\n")
    fh.writelines("echo \"CUDA devices visible: $CUDA_VISIBLE_DEVICES\"\n\n")

    fh.writelines("srun python inference.py\n\n")
    fh.writelines("echo \"Finished at $(date)\"\n")

os.system(f"sbatch {job_file}")
