import subprocess

logdir = "/home/chang_yuance/data/changyuance/codes/TFace-master_my/attribute/M3DFEL/outputs/M3DFEL_FiLM_center_loss_fold_1-[02-05]-[12:58]"
port = "6010"

cmd = [
    "tensorboard",
    f"--logdir={logdir}",
    f"--port={port}",
]

print("Running command:")
print(" ".join(cmd))

subprocess.run(cmd)