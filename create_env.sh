conda create -n tapex_repro_env python=3.8
source activate tapex_repro_env
pip install pip==23.3.2  
pip install fairseq==0.12.0
pip install hydra-core==1.0.7
pip install omegaconf==2.0.5
pip install tapex
pip install --editable ./
pip install seaborn
pip install pandas
pip install random
pip install json
pip install matplotlib
pip install numpy
