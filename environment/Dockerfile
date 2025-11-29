# hash:sha256:3b9f5ef778b2426c55a11e3aced5ba29cf8261828cbd15419c8a7672632e896a
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==3.0.14 \
        keras==2.2.4 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    numpy==1.19.4 \
    scikit-learn==0.22.1 \
    scipy==1.5.2 \
    tqdm==4.31.1

COPY postInstall /
RUN /postInstall
