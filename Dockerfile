FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# install dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y git python3-pip python3-opencv supervisor
RUN pip3 install --upgrade pip
RUN python3 -m pip install jupyterlab>=3.1 tensorboard jinja2
RUN python3 -m pip install -U setuptools setuptools_scm wheel

# configure supervisor
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# set up student user
RUN groupadd --gid 10042 student
RUN adduser --disabled-login --home /home/student --uid 10042 --gid 10042 student
USER student
RUN mkdir -p /home/student/.jupyter
COPY docker/jupyter_notebook_config.py /home/student/.jupyter

# install Python dependencies for notebooks
# the next simpler than trying to copy recursively with Docker's stupid ADD/COPY
RUN git clone https://github.com/fraunhofer-iais/namt-hackathon.git /home/student/hackathon
RUN pip3 --version
RUN pip3 install -r /home/student/hackathon/requirements.txt
RUN pip3 install /home/student/hackathon

# install special version of Torch/Torch Vision
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# fix for init bug with Torch
RUN pip install setuptools==59.5.0

# port configuration
EXPOSE 8888
EXPOSE 8889

# entry point
#ENTRYPOINT ["jupyter", "lab", "--collaborative", "--ip", "0.0.0.0"]
USER root
ENTRYPOINT ["/usr/bin/supervisord"]
