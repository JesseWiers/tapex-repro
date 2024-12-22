conda create -n tapex_repro_env python=3.8
source activate tapex_repro_env
pip install datasets
pip install pip==23.3.2  
pip install fairseq==0.12.0
pip install hydra-core==1.0.7
pip install omegaconf==2.0.5
cd Table-Pretraining-main
pip install tapex
pip install --editable ./

