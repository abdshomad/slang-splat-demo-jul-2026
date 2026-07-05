FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

# Avoid interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Vulkan and X11 libraries required for Vulkan loader
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libvulkan1 \
    vulkan-tools \
    libxext6 \
    libx11-6 \
    libgl1-mesa-glx \
    libglx-mesa0 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binaries from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency configuration files
COPY pyproject.toml uv.lock .python-version /app/

# Install the Python dependencies (uv will install Python 3.13 stand-alone)
RUN uv sync --frozen

# Copy the rest of the application code
COPY main.py index.html /app/
COPY slang-splat /app/slang-splat

# Expose port
EXPOSE 3000

# Run the FastAPI server using uv run
CMD ["uv", "run", "python", "main.py"]
