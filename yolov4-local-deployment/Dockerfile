#FROM ubuntu:20.04
FROM python:3

# Set a docker label to advertise multi-model support on the container
#LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
#LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true


#libGL.so.1 Need these for cv2
RUN apt update

#New Addition - Python with ubuntu
#RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev


#RUN \
#	apt-get install -y \
#	wget \
#	unzip \
#	ffmpeg \
#	git

#New Code End

RUN apt -y install libgl1-mesa-glx

#libgthread-2.0.so.0 Need this tool I guess for cv2
RUN apt-get -y install libgtk2.0-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
#RUN apt-get update && \
#    apt-get -y install --no-install-recommends \
#    build-essential \
#    ca-certificates \
#    openjdk-8-jdk-headless \
#    python3-dev \
#    curl \
#    vim \
#    && rm -rf /var/lib/apt/lists/* \
#    && curl -O https://bootstrap.pypa.io/get-pip.py \
#    && python3 get-pip.py

#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
#RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
#RUN pip3 --no-cache-dir install mxnet \
#                                multi-model-server \
#                                sagemaker-inference \
#                                retrying

COPY ./requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt
RUN pip3 install opencv-python --upgrade

#Copy the weights file inside docker
COPY ./pre_trained_model /home/pre_trained_model/

# Copy entrypoint script to the image
#COPY ./APP/entrypoint.py /usr/local/bin/dockerd-entrypoint.py
#RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

#RUN mkdir -p /home/model-server/

#Copy the default custom service file to handle incoming data and inference requests
#COPY ./APP/model_handler.py /home/model-server/model_handler.py
COPY ./APP/app.py /home/APP/app.py


#ENTRYPOINT ["python3" , "/usr/local/bin/dockerd-entrypoint.py"]
ENTRYPOINT ["python3" , "/home/APP/app.py"]
