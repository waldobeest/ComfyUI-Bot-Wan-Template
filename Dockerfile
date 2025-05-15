# Use multi-stage build with caching optimizations
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

# ------------------------------------------------------------
# Consolidated environment variables
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# ------------------------------------------------------------
# System packages + Python 3.12 venv
# ------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        curl ffmpeg ninja-build git git-lfs wget vim \
        libgl1 libglib2.0-0 build-essential gcc && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3.12 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# ------------------------------------------------------------
# PyTorch (CUDA 12.8) & core tooling (no pip cache mounts)
# ------------------------------------------------------------
# 2) Install PyTorch (CUDA 12.8) & freeze torch versions to constraints file
RUN pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    # Save exact installed torch versions
    pip freeze | grep -E "^(torch|torchvision|torchaudio)" > /tmp/torch-constraint.txt && \
    # Install core tooling
    pip install packaging setuptools wheel pyyaml gdown triton runpod opencv-python

# 3) Clone ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /ComfyUI

# 4) Install ComfyUI requirements using torch constraint file
RUN cd /ComfyUI && \
    pip install -r requirements.txt --constraint /tmp/torch-constraint.txt

# ------------------------------------------------------------
# Final stage
# ------------------------------------------------------------
# ------------------------------------------------------------
# Final stage
# ------------------------------------------------------------
FROM base AS final
ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir -p /models/diffusion_models /models/text_encoders /models/vae /models/clip_vision

# Create LoRA directory and download LoRA files
# Create the directory for models
RUN mkdir -p /models/loras

# Download all files in a single layer to reduce image size
RUN cd /models/loras && \
    wget -O "wan-nsfw-e14-fixed.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/wan-nsfw-e14-fixed.safetensors" && \
    wget -O "big_tits_epoch_50.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/big_tits_epoch_50.safetensors" && \
    wget -O "pov_blowjob_v1.1.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/pov_blowjob_v1.1.safetensors" && \
    wget -O "Wan_Breast_Helper_Hearmeman.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/Wan_Breast_Helper_Hearmeman.safetensors" && \
    wget -O "wan_cowgirl_v1.3.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/wan_cowgirl_v1.3.safetensors" && \
    wget -O "cleavage_epoch_40.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/cleavage_epoch_40.safetensors" && \
    wget -O "orgasm_e60.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/orgasm_e60.safetensors" && \
    wget -O "wan_missionary_side.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/wan_missionary_side.safetensors" && \
    wget -O "dicks_epoch_100.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/dicks_epoch_100.safetensors" && \
    wget -O "masturbation_cumshot_wanI2V480p_v1.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/masturbation_cumshot_wanI2V480p_v1.safetensors" && \
    wget -O "r0und4b0ut-wan-v1.0.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/r0und4b0ut-wan-v1.0.safetensors" && \
    wget -O "facials_epoch_50.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/facials_epoch_50.safetensors" && \
    wget -O "deepthroat_epoch_80.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/deepthroat_epoch_80.safetensors" && \
    wget -O "ahegao_v1_e35_wan.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/ahegao_v1_e35_wan.safetensors" && \
    wget -O "Wan_Pussy_LoRA_Hearmeman.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/Wan_Pussy_LoRA_Hearmeman.safetensors" && \
    wget -O "doggyPOV_v1_1.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/doggyPOV_v1_1.safetensors" && \
    wget -O "wan_pov_missionary_v1.1.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/wan_pov_missionary_v1.1.safetensors" && \
    wget -O "Titfuck_WAN14B_V1_Release.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/Titfuck_WAN14B_V1_Release.safetensors" && \
    wget -O "FILM_NOIR_EPOCH10.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/FILM_NOIR_EPOCH10.safetensors" && \
    wget -O "BouncyWalkV01.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/BouncyWalkV01.safetensors" && \
    wget -O "Spinning%20V2.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/Spinning%20V2.safetensors" && \
    wget -O "squish_18.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/squish_18.safetensors" && \
    wget -O "detailz-wan.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/detailz-wan.safetensors" && \
    wget -O "studio_ghibli_wan14b_t2v_v01.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/studio_ghibli_wan14b_t2v_v01.safetensors" && \
    wget -O "Su_Bl_Ep02-Wan.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/Su_Bl_Ep02-Wan.safetensors" && \
    wget -O "wan_female_masturbation.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/wan_female_masturbation.safetensors" && \
    wget -O "Wan-Hip_Slammin_Assertive_Cowgirl.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/Wan-Hip_Slammin_Assertive_Cowgirl.safetensors" && \
    wget -O "T2V%20-%20Skinny%20Petite%20Instagram%20Women%20-%2014B.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/T2V%20-%20Skinny%20Petite%20Instagram%20Women%20-%2014B.safetensors" && \
    wget -O "T2V-jiggle_tits-14b.safetensors" "https://d1s3da0dcaf6kx.cloudfront.net/T2V-jiggle_tits-14b.safetensors"



# Download frame interpolation checkpoint
RUN mkdir -p /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/film && \
    wget -O /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/film/film_net_fp32.pt \
        https://d1s3da0dcaf6kx.cloudfront.net/film_net_fp32.pt

# Split diffusion model downloads to avoid 50GB+ layers
RUN wget -P /models/diffusion_models https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors
RUN wget -P /models/diffusion_models https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors
RUN wget -P /models/diffusion_models https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_preview_fp16.safetensors
RUN wget -P /models/diffusion_models https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors

# Split text encoders
RUN wget -P /models/text_encoders https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors
RUN wget -P /models/text_encoders https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors
RUN wget -P /models/text_encoders https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# Split VAE downloads
RUN wget -P /models/vae https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors
RUN wget -P /models/vae https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors

# Clip vision
RUN wget -P /models/clip_vision https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors




RUN pip install opencv-python

RUN git clone https://github.com/Hearmeman24/upscalers.git /tmp/upscalers \
    && cp /tmp/upscalers/4xLSDIR.pth /4xLSDIR.pth \
    && rm -rf /tmp/upscalers

# Clone and install all your custom nodes
RUN for repo in \
    https://github.com/kijai/ComfyUI-KJNodes.git \
    https://github.com/Comfy-Org/ComfyUI-Manager.git \
    https://github.com/nonnonstop/comfyui-faster-loading.git \
    https://github.com/rgthree/rgthree-comfy.git \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    https://github.com/cubiq/ComfyUI_essentials.git \
    https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    https://github.com/chrisgoringe/cg-use-everywhere.git \
    https://github.com/tsogzark/ComfyUI-load-image-from-url.git; \
  do \
    cd /ComfyUI/custom_nodes; \
    repo_dir=$(basename "$repo" .git); \
    git clone "$repo"; \
    if [ -f "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" ]; then \
      pip install -r "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" --constraint /torch-constraint.txt; \
    fi; \
    if [ -f "/ComfyUI/custom_nodes/$repo_dir/install.py" ]; then \
      python "/ComfyUI/custom_nodes/$repo_dir/install.py"; \
    fi; \
  done


RUN pip install --no-cache-dir \
    https://raw.githubusercontent.com/Hearmeman24/upscalers/master/sageattention-2.1.1-cp312-cp312-linux_x86_64.whl

RUN pip install --no-cache-dir discord.py==2.5.2 \
                              python-dotenv==1.1.0 \
                              Requests==2.32.3 \
                              websocket_client==1.8.0 \
                              "httpx[http2]"

# Entrypointtt
COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh
EXPOSE 8888
CMD ["/start_script.sh"]
