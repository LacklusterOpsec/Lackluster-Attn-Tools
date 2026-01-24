"""
Persistent cache system for benchmark results.
"""
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path

CACHE_FILE = Path(__file__).parent / "benchmark_db.json"
CACHE_VERSION = 1


def load_cache():
    """Load cache from disk."""
    if not CACHE_FILE.exists():
        return {"version": CACHE_VERSION, "results": {}}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("version", 0) != CACHE_VERSION:
                # Invalidate old cache versions
                return {"version": CACHE_VERSION, "results": {}}
            return data
    except (json.JSONDecodeError, IOError):
        return {"version": CACHE_VERSION, "results": {}}


def save_cache(cache_data):
    """Save cache to disk."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[Benchmark] Warning: Could not save cache: {e}")


def get_cached_result(cache_key):
    """Get a cached result by key."""
    cache = load_cache()
    return cache["results"].get(cache_key)


def set_cached_result(cache_key, result):
    """Store a result in cache."""
    cache = load_cache()
    result["timestamp"] = datetime.now().isoformat()
    cache["results"][cache_key] = result
    save_cache(cache)


def get_model_hash(model):
    """
    Generate a unique hash for a model based on its parameters and config.
    """
    try:
        # Get model's internal model object
        inner_model = getattr(model, 'model', model)

        # Try to get model config
        config_str = ""
        if hasattr(inner_model, 'model_config'):
            config_str = str(inner_model.model_config)
        elif hasattr(inner_model, 'config'):
            config_str = str(inner_model.config)

        # Get first few parameter shapes for uniqueness
        param_info = []
        params = None
        if hasattr(inner_model, 'parameters'):
            params = list(inner_model.parameters())
        elif hasattr(inner_model, 'diffusion_model') and hasattr(inner_model.diffusion_model, 'parameters'):
            params = list(inner_model.diffusion_model.parameters())

        if params:
            for p in params[:5]:
                param_info.append(str(p.shape))
                param_info.append(str(p.dtype))

        hash_input = config_str + "".join(param_info)
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    except Exception as e:
        # Fallback: use id of model object
        return hashlib.md5(str(id(model)).encode()).hexdigest()[:16]


def get_model_dtype(model):
    """Get the dtype of a model."""
    try:
        inner_model = getattr(model, 'model', model)

        # Try different ways to get parameters
        params = None
        if hasattr(inner_model, 'parameters'):
            params = inner_model.parameters()
        elif hasattr(inner_model, 'diffusion_model') and hasattr(inner_model.diffusion_model, 'parameters'):
            params = inner_model.diffusion_model.parameters()

        if params:
            first_param = next(iter(params), None)
            if first_param is not None:
                dtype = str(first_param.dtype)
                # Clean up dtype string
                return dtype.replace("torch.", "")

        return "unknown"
    except Exception:
        return "unknown"


def get_head_dim(model):
    """
    Extract head dimension from model architecture.
    Common values: 64 (SD1.5), 128 (SDXL), 160 (LTX)
    """
    try:
        inner_model = getattr(model, 'model', model)

        # Check model config
        config = None
        if hasattr(inner_model, 'model_config'):
            config = inner_model.model_config
        elif hasattr(inner_model, 'config'):
            config = inner_model.config

        if config:
            # Try various config attributes
            if hasattr(config, 'unet_config'):
                unet_config = config.unet_config
                if isinstance(unet_config, dict):
                    # Check for num_head_channels or similar
                    if 'num_head_channels' in unet_config:
                        return unet_config['num_head_channels']
                    if 'attention_head_dim' in unet_config:
                        head_dim = unet_config['attention_head_dim']
                        if isinstance(head_dim, list):
                            return head_dim[0]
                        return head_dim

        # Try to find attention modules and extract head_dim
        diffusion_model = None
        if hasattr(inner_model, 'diffusion_model'):
            diffusion_model = inner_model.diffusion_model
        elif hasattr(inner_model, 'model') and hasattr(inner_model.model, 'diffusion_model'):
            diffusion_model = inner_model.model.diffusion_model

        if diffusion_model:
            for name, module in diffusion_model.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    if hasattr(module, 'head_dim'):
                        return module.head_dim
                    if hasattr(module, 'heads') and hasattr(module, 'inner_dim'):
                        return module.inner_dim // module.heads

        # Default fallback
        return 128

    except Exception:
        return 128


def build_cache_key(model_hash, dtype, head_dim):
    """Build a cache key from model properties."""
    return f"{model_hash}_{dtype}_{head_dim}"
