wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh

#https://download.pytorch.org/whl/torch_stable.html
wget https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp39-cp39-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp39-cp39-linux_x86_64.whl

pip install pillow  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.9.0  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision-0.10.0+cu111-cp39-cp39-linux_x86_64.whl
pip install deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple