"""Qwen Image 2.1 through the locally installed ComfyUI HTTP API."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import tempfile
import time
import urllib.parse
import urllib.request

BASE_URL = os.environ.get('IMAGE_BASE_URL', 'http://127.0.0.1:8188').rstrip('/')
COMFY_DIR = Path(os.environ.get('COMFYUI_DIR', str(Path.home() / 'ComfyUI-standalone')))
TIMEOUT = int(os.environ.get('IMAGE_TIMEOUT', '240'))
_lock = threading.Lock()
_process = None


def api(path, data=None, timeout=10):
    request = urllib.request.Request(BASE_URL + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def ensure_server():
    global _process
    try:
        api('/system_stats', timeout=2)
        return
    except (OSError, ValueError):
        pass
    local = urllib.parse.urlsplit(BASE_URL)
    if local.hostname != '127.0.0.1' or os.environ.get('IMAGE_AUTOSTART', '1') != '1':
        raise RuntimeError('로컬 이미지 서버에 연결할 수 없어요.')
    python = COMFY_DIR / '.venv/bin/python'
    if not python.exists() or not (COMFY_DIR / 'main.py').exists():
        raise RuntimeError('ComfyUI 설치 경로를 COMFYUI_DIR로 지정해 주세요.')
    if _process is None or _process.poll() is not None:
        log_path = Path(__file__).resolve().parent.parent / 'image-model.log'
        with log_path.open('ab') as log:
            _process = subprocess.Popen([str(python), str(COMFY_DIR / 'main.py'), '--listen', '127.0.0.1',
                                         '--port', str(local.port or 8188), '--disable-auto-launch',
                                         '--disable-all-custom-nodes'], cwd=COMFY_DIR, stdout=log, stderr=log,
                                        start_new_session=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if _process.poll() is not None:
            raise RuntimeError('ComfyUI 시작에 실패했어요. image-model.log를 확인해 주세요.')
        try:
            api('/system_stats', timeout=2)
            return
        except (OSError, ValueError):
            time.sleep(1)
    raise TimeoutError('로컬 이미지 서버 시작 시간이 초과됐어요.')


def workflow(prompt, width=432, height=768):
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': os.environ.get('IMAGE_MODEL', 'qwen_image_2.1_int8_convrot.safetensors'), 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': os.environ.get('IMAGE_TEXT_ENCODER', 'qwen3vl_8b_int8_convrot.safetensors'), 'type': 'qwen_image', 'device': 'default'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': os.environ.get('IMAGE_VAE', 'qwen_image_2.1_vae_bf16.safetensors')}},
        '4': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['2', 0], 'prompt': prompt, 'negative_prompt': 'letters, words, watermark, logo, blurry, distorted', 'resolution': 256}},
        '5': {'class_type': 'EmptyLatentImage', 'inputs': {'width': width, 'height': height, 'batch_size': 1}},
        '6': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'seed': 20260930, 'steps': 8, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['4', 0], 'negative': ['4', 1], 'latent_image': ['5', 0], 'denoise': 1.0}},
        '7': {'class_type': 'VAEDecode', 'inputs': {'samples': ['6', 0], 'vae': ['3', 0]}},
        '8': {'class_type': 'SaveImage', 'inputs': {'images': ['7', 0], 'filename_prefix': 'hey-alex'}},
    }


def generate(prompt, destination, *, width=432, height=768):
    # Serialize GPU work; never hold up the conversation's SSE response.
    with _lock:
        ensure_server()
        response = api('/prompt', {'prompt': workflow(prompt, width, height)})
        prompt_id = response['prompt_id']
        deadline = time.monotonic() + TIMEOUT
        try:
            while time.monotonic() < deadline:
                entry = api('/history/' + prompt_id).get(prompt_id)
                if entry:
                    if entry.get('status', {}).get('status_str') == 'error':
                        raise RuntimeError('로컬 이미지 모델 실행에 실패했어요.')
                    for output in entry.get('outputs', {}).values():
                        for picture in output.get('images', []):
                            query = urllib.parse.urlencode(picture)
                            with urllib.request.urlopen(BASE_URL + '/view?' + query, timeout=30) as result:
                                content = result.read(20 * 1024 * 1024 + 1)
                            if not content.startswith(b'\x89PNG\r\n\x1a\n') or len(content) > 20 * 1024 * 1024:
                                raise ValueError('이미지 서버가 올바른 PNG를 반환하지 않았어요.')
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as output:
                                output.write(content)
                            Path(output.name).replace(destination)
                            return
                time.sleep(1)
            raise TimeoutError('이미지 생성 시간이 초과됐어요.')
        finally:
            # Remove only our queued task; do not interrupt other ComfyUI users.
            try:
                api('/queue', {'delete': [prompt_id]})
            except (OSError, ValueError):
                pass
