import os
import sys
import numpy as np

#PBS -l select=1:ncpus=28:mem=120gb:ngpus=2:gpu_model=p100:interconnect=fdr,walltime=4:00:00
#PBS -l select=1:ncpus=40:mem=370gb:ngpus=2:gpu_model=v100:interconnect=hdr,walltime=4:00:00

batch_size = 64
num_units = 16
epochs = 100

pbsfile = open("job.pbs", "w")

pbsfile.write("""#PBS -u apatyk
#PBS -N DP_E{:d}
#PBS -l select=1:ncpus=40:mem=370gb:ngpus=2:gpu_model=v100:interconnect=hdr,walltime=16:00:00
#PBS -m ae
#PBS -o /home/apatyk/Research/NewModels/results-gru-b{:d}-u{:d}-e{:d}.txt
#PBS -j oe

""".format(epochs,batch_size,num_units,epochs))

pbsfile.write("""ulimit -c 0
source /software/spackages/linux-centos8-x86_64/gcc-8.3.1/anaconda3-2019.10-v5cuhr6keyz5ryxcwvv2jkzfj2gwrj4a/etc/profile.d/conda.sh
module load cuda/10.2.89-gcc/8.3.1 cudnn/8.0.0.180-10.2-linux-x64-gcc/8.3.1 anaconda3/2019.10-gcc/8.3.1
conda activate tf_env
cd  ~/Research/NewModels/ 
python DailyPatternLSTM.py {:d} {:d} {:d}
""".format(batch_size,num_units,epochs))
pbsfile.write("""if [ $? -ne 0 ]; then
rm core.*
fi
exit""")

pbsfile.close()
os.system("qsub job.pbs")
