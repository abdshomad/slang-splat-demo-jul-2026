import sys
import os
import time
import io
import threading
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add slang-splat subdirectory to Python path to import its modules
REPO_ROOT = Path(__file__).resolve().parent / "slang-splat"
sys.path.insert(0, str(REPO_ROOT))

from src.app.cli import _init_hparams, _training_params
from src.app.shared import apply_training_profile, build_training_params, estimate_scene_bounds, SceneBounds
from src.renderer import Camera, GaussianRenderSettings, GaussianRenderer
from src.scene import (
    GaussianInitHyperParams,
    build_training_frames,
    initialize_scene_from_colmap_points,
    load_colmap_reconstruction,
    resolve_colmap_init_hparams
)
from src.training import GaussianTrainer, resolve_effective_train_render_factor, resolve_training_resolution
from src import create_default_device

# Initialize FastAPI app
app = FastAPI(title="Slang-Splat Web Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for training state
class TrainingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.device = None
        self.recon = None
        self.frames = None
        self.renderer = None
        self.scene = None
        self.trainer = None
        self.scene_bounds = None
        
        # Training loop state
        self.is_training = False
        self.current_step = 0
        self.max_iters = 1000
        self.loss = 0.0
        self.avg_loss = 0.0
        self.last_psnr = 0.0
        self.avg_psnr = 0.0
        self.num_gaussians = 0
        self.elapsed_time = 0.0
        self.start_time = 0.0
        self.stop_requested = False
        self.training_thread = None
        self.history = []
        
        # Hyperparameters
        self.dataset_root = REPO_ROOT / "dataset" / "garden"
        self.images_subdir = "images_4"
        self.max_gaussians_limit = 5000
        self.initial_opacity = 0.5
        self.init_mode = "colmap" # "colmap" or "diffused"
        self.width = 256
        self.height = 256
        self.training_profile = "legacy"
        
    def initialize_scene(self):
        with self.lock:
            if self.device is None:
                self.device = create_default_device(enable_debug_layers=False)
                
            # Load COLMAP reconstruction
            if self.recon is None:
                self.recon = load_colmap_reconstruction(self.dataset_root, sparse_subdir="sparse/0")
            
            self.frames = build_training_frames(self.recon, images_subdir=self.images_subdir)
            
            # Setup params via actual CLI parser
            from src.app.cli import build_parser
            parser = build_parser()
            args = parser.parse_args(["train-colmap", "--colmap-root", str(self.dataset_root)])
            
            # Override options with our custom interactive settings
            args.max_gaussians = self.max_gaussians_limit
            args.init_opacity = self.initial_opacity
            args.width = self.width
            args.height = self.height
            args.iters = self.max_iters
            args.training_profile = self.training_profile
            args.use_sh = False  # Keep it simple and fast for demo
            
            init_hparams = _init_hparams(args)
            if self.init_mode == "diffused":
                # Jitter diffused settings
                from dataclasses import replace
                init_hparams = replace(init_hparams, position_jitter_std=0.01)
                
            params, profile = apply_training_profile(
                _training_params(args),
                self.training_profile,
                dataset_root=self.dataset_root,
                images_subdir=self.images_subdir
            )
            
            # Opacity override
            init_hparams = replace_opacity(init_hparams, self.initial_opacity)
            
            resolved_init = resolve_colmap_init_hparams(self.recon, params.training.max_gaussians, init_hparams)
            
            self.scene = initialize_scene_from_colmap_points(
                recon=self.recon,
                max_gaussians=params.training.max_gaussians,
                seed=42,
                init_hparams=resolved_init
            )
            
            # Create renderer
            from src.renderer.render_params import RendererParams
            from src.app.cli import _CLI_COMMON_RENDER_DEFAULTS
            renderer_params = RendererParams.from_args(args, _CLI_COMMON_RENDER_DEFAULTS)
            settings = GaussianRenderSettings.from_renderer_params(self.width, self.height, renderer_params)
            self.renderer = settings.create_renderer(self.device)
            
            # Setup trainer
            self.trainer = GaussianTrainer(
                device=self.device,
                renderer=self.renderer,
                scene=self.scene,
                frames=self.frames,
                adam_hparams=params.adam,
                stability_hparams=params.stability,
                training_hparams=params.training,
                seed=42,
                scale_reg_reference=float(max(resolved_init.base_scale, 1e-8)),
            )
            
            self.scene_bounds = estimate_scene_bounds(self.scene)
            
            # Reset loop state
            self.current_step = 0
            self.loss = 0.0
            self.avg_loss = 0.0
            self.last_psnr = 0.0
            self.avg_psnr = 0.0
            self.num_gaussians = self.scene.count
            self.elapsed_time = 0.0
            self.history = []
            self.stop_requested = False

def replace_opacity(hparams, opacity):
    # GaussianInitHyperParams is frozen, so we reconstruct it or return updated copy
    from dataclasses import replace
    return replace(hparams, initial_opacity=opacity)

state = TrainingState()

def training_thread_loop():
    global state
    state.start_time = time.perf_counter() - state.elapsed_time
    
    while not state.stop_requested and state.current_step < state.max_iters:
        try:
            with state.lock:
                loss = state.trainer.step()
                state.current_step += 1
                state.loss = float(loss)
                state.avg_loss = float(state.trainer.state.avg_loss)
                state.last_psnr = float(state.trainer.state.last_psnr)
                state.avg_psnr = float(state.trainer.state.avg_psnr)
                state.num_gaussians = int(state.trainer.scene.count)
                state.elapsed_time = time.perf_counter() - state.start_time
                
                # Append to history periodically to save space
                if state.current_step == 1 or state.current_step % 5 == 0:
                    state.history.append({
                        "step": state.current_step,
                        "loss": state.loss,
                        "avg_loss": state.avg_loss,
                        "psnr": state.last_psnr,
                        "avg_psnr": state.avg_psnr,
                        "gaussians": state.num_gaussians
                    })
            
            # Yield/sleep a tiny bit to allow rendering threads to access the GPU/lock
            time.sleep(0.002)
        except Exception as e:
            print(f"Error in training step: {e}", file=sys.stderr)
            state.is_training = False
            break
            
    state.is_training = False

class InitParamsModel(BaseModel):
    max_gaussians: int = 5000
    initial_opacity: float = 0.5
    init_mode: str = "colmap" # "colmap" or "diffused"
    width: int = 256
    height: int = 256
    max_iters: int = 1000

@app.post("/api/initialize")
def initialize_scene(params: InitParamsModel):
    try:
        state.max_gaussians_limit = params.max_gaussians
        state.initial_opacity = params.initial_opacity
        state.init_mode = params.init_mode
        state.width = params.width
        state.height = params.height
        state.max_iters = params.max_iters
        
        state.initialize_scene()
        
        return {
            "status": "initialized",
            "gaussians": state.num_gaussians,
            "bounds": {
                "center": state.scene_bounds.center.tolist(),
                "radius": float(state.scene_bounds.radius)
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start")
def start_training():
    if state.scene is None:
        raise HTTPException(status_code=400, detail="Scene not initialized. Call /api/initialize first.")
        
    if state.is_training:
        return {"status": "already_training"}
        
    state.is_training = True
    state.stop_requested = False
    state.training_thread = threading.Thread(target=training_thread_loop, daemon=True)
    state.training_thread.start()
    
    return {"status": "started"}

@app.post("/api/stop")
def stop_training():
    if not state.is_training:
        return {"status": "not_training"}
        
    state.stop_requested = True
    # Wait for thread to finish
    if state.training_thread:
        state.training_thread.join(timeout=1.0)
    state.is_training = False
    return {"status": "stopped"}

@app.post("/api/reset")
def reset_scene():
    # Stop first
    if state.is_training:
        state.stop_requested = True
        if state.training_thread:
            state.training_thread.join(timeout=1.0)
        state.is_training = False
        
    state.initialize_scene()
    return {"status": "reset", "gaussians": state.num_gaussians}

@app.get("/api/status")
def get_status():
    return {
        "is_training": state.is_training,
        "current_step": state.current_step,
        "max_iters": state.max_iters,
        "loss": state.loss,
        "avg_loss": state.avg_loss,
        "psnr": state.last_psnr,
        "avg_psnr": state.avg_psnr,
        "gaussians": state.num_gaussians,
        "elapsed_time": state.elapsed_time,
        "history": state.history
    }

@app.get("/api/frames")
def get_frames():
    if state.frames is None:
        try:
            state.initialize_scene()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    return {
        "count": len(state.frames),
        "frames": [{"index": i, "width": int(f.width), "height": int(f.height)} for i, f in enumerate(state.frames)]
    }

@app.get("/api/render/frame/{frame_idx}")
def render_frame(frame_idx: int):
    if state.scene is None or state.renderer is None or state.frames is None:
        raise HTTPException(status_code=400, detail="Scene not initialized.")
        
    if frame_idx < 0 or frame_idx >= len(state.frames):
        raise HTTPException(status_code=404, detail="Frame index out of range.")
        
    frame = state.frames[frame_idx]
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    with state.lock:
        state.renderer.set_scene(state.scene)
        tex, _ = state.renderer.render_to_texture(
            frame.make_camera(near=0.1, far=100.0),
            background=background
        )
        rgba = tex.to_numpy()
        
    # Convert RGBA numpy array to JPEG
    rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
    # Flip Y as slang-splat views are flipped in save_snapshot
    img = Image.fromarray((255.0 * np.flipud(rgb) + 0.5).astype(np.uint8), mode="RGB")
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

@app.get("/api/render/orbit")
def render_orbit(
    azimuth: float = Query(0.0, description="Azimuth angle in degrees"),
    elevation: float = Query(20.0, description="Elevation angle in degrees"),
    distance_mult: float = Query(1.0, description="Distance multiplier relative to scene radius"),
    fov: float = Query(60.0, description="Field of view in degrees")
):
    if state.scene is None or state.renderer is None:
        raise HTTPException(status_code=400, detail="Scene not initialized.")
        
    bounds = state.scene_bounds
    center = bounds.center
    radius = bounds.radius
    distance = max(distance_mult * radius, 0.1)
    
    az = np.radians(azimuth)
    el = np.radians(elevation)
    
    # Calculate position on sphere centered at the scene bounds center
    x = center[0] + distance * np.cos(el) * np.sin(az)
    y = center[1] + distance * np.sin(el)
    z = center[2] - distance * np.cos(el) * np.cos(az)
    
    position = np.array([x, y, z], dtype=np.float32)
    target = np.array(center, dtype=np.float32)
    
    # Correct up vector
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if np.abs(np.cos(el)) < 1e-4:
        up = np.array([0.0, 0.0, 1.0] if elevation > 0 else [0.0, 0.0, -1.0], dtype=np.float32)
        
    camera = Camera.look_at(
        position=position,
        target=target,
        up=up,
        fov_y_degrees=fov,
        near=0.1,
        far=distance + 4.0 * radius
    )
    
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    with state.lock:
        state.renderer.set_scene(state.scene)
        tex, _ = state.renderer.render_to_texture(camera, background=background)
        rgba = tex.to_numpy()
        
    rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
    img = Image.fromarray((255.0 * np.flipud(rgb) + 0.5).astype(np.uint8), mode="RGB")
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

# Serve index.html directly
@app.get("/")
def get_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    # Initialize default scene on launch
    try:
        print("Pre-initializing default scene...")
        state.initialize_scene()
        print("Pre-initialization complete.")
    except Exception as e:
        print(f"Error during pre-initialization: {e}", file=sys.stderr)
        
    uvicorn.run(app, host="0.0.0.0", port=3000)
