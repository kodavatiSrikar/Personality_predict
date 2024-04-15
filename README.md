# Personality_predict

Setup a pytorch conda enviroment:
```
conda create -n pytorch_env -c pytorch pytorch torchvision
```
activate the environment:
```
conda activate pytorch_env
```
install all the requirments:
```
pip install -r requirements.txt
```
Download the dataset in the Personality_predict folder:

https://drive.google.com/file/d/1IbMMKZlZpo3LCKHR7kNUNo-ZMynDjmaI/view?usp=sharing

Run CPU version of code:
```
python pers_cpu.py
```
Run GPU version of code:
```
python pers_gpu.py
```
