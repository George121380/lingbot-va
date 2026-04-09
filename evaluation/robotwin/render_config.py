import os


SUPPORTED_DENOISERS = {"none", "oidn", "optix"}


def configure_sapien_ray_tracing(default_denoiser: str = "optix") -> None:
    """Patch SAPIEN's denoiser selection for this process.

    The current runtime logs repeated OIDN CUDA errors on this machine.
    Defaulting to `optix` avoids those backend errors while preserving
    denoising on NVIDIA GPUs, and still allows an explicit override via
    `SAPIEN_RT_DENOISER`.
    """
    try:
        import sapien.core as sapien
    except Exception:
        return

    render = sapien.render
    if getattr(render, "_lingbot_safe_denoiser_patched", False):
        return

    default = default_denoiser.lower()
    if default not in SUPPORTED_DENOISERS:
        default = "optix"

    env_choice = os.environ.get("SAPIEN_RT_DENOISER", "").strip().lower()
    if env_choice not in SUPPORTED_DENOISERS:
        env_choice = ""

    original = render.set_ray_tracing_denoiser

    def safe_set_ray_tracing_denoiser(name: str) -> None:
        requested = (name or "").strip().lower()
        target = env_choice or requested or default

        if target not in SUPPORTED_DENOISERS:
            target = default

        # OIDN is the source of repeated CUDA backend errors on this setup.
        if target == "oidn" and not env_choice:
            target = default

        original(target)

    render.set_ray_tracing_denoiser = safe_set_ray_tracing_denoiser
    render._lingbot_safe_denoiser_patched = True
