import subprocess

for i in range(2, 11):  # 1 to 10 inclusive
    input_file = f"/home/socio/ysocial-simulations/MADOC/reddit-technology/sample_{i}/technology_sample_{i}.parquet"
    cmd = [
        "nohup",
        "python3",
        "core-periphery-network-real.py",
        input_file
    ]
    # Run each job in background and redirect output
    with open(f"job_{i}.out", "w") as outfile, open(f"job_{i}.err", "w") as errfile:
        subprocess.Popen(cmd, stdout=outfile, stderr=errfile)

print ('Done submitting jobs')