import os

#PBS -l select=1:ncpus=28:mem=120gb:ngpus=2:gpu_model=p100:interconnect=fdr,walltime=4:00:00
#PBS -l select=1:ncpus=24:mem=120gb:ngpus=2:gpu_model=k40:interconnect=fdr,walltime=4:00:00
#PBS -l select=1:ncpus=40:mem=370gb:ngpus=2:gpu_model=v100:interconnect=hdr,walltime=72:00:00
#PBS -l select=1:ncpus=56:mem=370gb:ngpus=2:gpu_model=v100s:interconnect=hdr,walltime=72:00:00

#window_lengths = [2, 4, 6, 8, 10]
window_lengths = [6]

for W in window_lengths:

    pbsfile = open("job.pbs", "w")

    pbsfile.write("""#PBS -u apatyk
    #PBS -N GenSamples_{:d}Min
    #PBS -l select=1:ncpus=40:mem=370gb:ngpus=2:gpu_model=v100:interconnect=hdr,walltime=72:00:00
    #PBS -m ae
    #PBS -o /home/apatyk/Research/GenerateSamples/results/gen-samples-{:d}min.txt
    #PBS -j oe

    """.format(W, W))

    pbsfile.write("""ulimit -c 0
    source /software/spackages/linux-centos8-x86_64/gcc-8.3.1/anaconda3-2019.10-v5cuhr6keyz5ryxcwvv2jkzfj2gwrj4a/etc/profile.d/conda.sh
    module load cuda/10.2.89-gcc/8.3.1 cudnn/8.0.0.180-10.2-linux-x64-gcc/8.3.1 anaconda3/2019.10-gcc/8.3.1
    conda activate tf_env
    cd  ~/Research/GenerateSamples/ 
    python GenerateSamples.py {:d}
    """.format(W))
    pbsfile.write("""if [ $? -ne 0 ]; then
    rm core.*
    fi
    exit""")

    pbsfile.close()
    os.system("qsub job.pbs")
