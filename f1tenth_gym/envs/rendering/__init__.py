import os
from typing import Optional, TYPE_CHECKING

from .renderer import RenderSpec, EnvRenderer, ObjectRenderer
from .callbacks import make_lidar_scan_callback
from ..dynamic_models import VehicleParameters

if TYPE_CHECKING:
    from ..track import Track

__all__ = [
    "RenderSpec",
    "EnvRenderer",
    "ObjectRenderer",
    "make_lidar_scan_callback",
    "make_renderer",
]

def _use_gl_for_offscreen(method: str) -> bool:
    """Decide whether rgb_array frames come from the fast GL framebuffer grab.

    The GL widget cannot render under the headless ``offscreen`` Qt platform
    (it has no FBO there), so a fast GL grab needs a real X server or a virtual
    one (``xvfb-run``), signalled by ``$DISPLAY``. Without a display we fall
    back to the 2D raster exporter, which works headless with zero setup
    (e.g. Colab without xvfb).

    * ``"gl"``  -> always GL (caller guarantees a display).
    * ``"2d"``  -> always the 2D raster exporter.
    * ``"auto"``-> GL if ``$DISPLAY`` is set, else 2D.
    """
    if method == "gl":
        return True
    if method == "2d":
        return False
    return bool(os.environ.get("DISPLAY"))


def make_renderer(
    params: VehicleParameters,
    track: "Track",
    agent_ids: list[str],
    render_mode: Optional[str] = None,
    render_fps: Optional[int] = 60,
    render_spec: RenderSpec = RenderSpec(),
) -> tuple[EnvRenderer, RenderSpec]:
    """Return an instance of the renderer and the rendering specification."""
    is_rgb = render_mode in ("rgb_array", "rgb_array_list")
    method = getattr(render_spec, "frame_output_method", "auto")

    if render_spec.render_type == "pyqt6":
        if is_rgb:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from .rendering_pyqt import PyQtEnvRenderer as EnvRenderer
    elif render_spec.render_type == "pyqt6gl":
        if is_rgb:
            if _use_gl_for_offscreen(method):
                # fast GL framebuffer grab; needs a display (real X or xvfb-run)
                os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
                from .rendering_pyqtgl import PyQtEnvRendererGL as EnvRenderer
            else:
                # headless, zero-setup fallback: 2D raster exporter
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
                from .rendering_pyqt import PyQtEnvRenderer as EnvRenderer
        else:
            from .rendering_pyqtgl import PyQtEnvRendererGL as EnvRenderer
    else:
        raise ValueError(f"Unknown render type: {render_spec.render_type}")

    if render_mode in ["human", "rgb_array", "unlimited", "human_fast"]:
        renderer = EnvRenderer(
            params=params,
            track=track,
            agent_ids=agent_ids,
            render_spec=render_spec,
            render_mode=render_mode,
            render_fps=render_fps,
        )
    else:
        renderer = None
    return renderer, render_spec
