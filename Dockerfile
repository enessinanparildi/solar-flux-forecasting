# TensorRT + CUDA base — matches TRT used to build the engine locally
FROM nvcr.io/nvidia/tensorrt:24.08-py3

WORKDIR /opt/ml/code

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and entrypoint
COPY build_trt_engine.py serve.py serve_ray.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Copy model artifacts
# ONNX is always required. TRT engine is optional — if absent it will be
# built at container startup on the target GPU (required for SageMaker
# since engines are GPU-architecture specific).
COPY models/ ./models/

# SageMaker requires port 8080
EXPOSE 8080

# Build TRT engine if needed, then start the server
ENTRYPOINT ["./entrypoint.sh"]