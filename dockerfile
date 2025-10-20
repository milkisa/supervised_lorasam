FROM nvcr.io/nvidia/pytorch:22.04-py3
COPY requirements.txt .
RUN pip install monai
RUN pip install transformers
RUN pip install Pillow>=9.1.0
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install git+https://github.com/openai/CLIP.git