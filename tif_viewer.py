"""Convenience launcher for the missing-struts interactive TIFF viewer."""

from src.visual_engine import DEFAULT_INPUT, open_interactive_viewer


if __name__ == "__main__":
    info = open_interactive_viewer(DEFAULT_INPUT)
    print(
        f"Loaded {info.path.name}: shape={info.shape}, "
        f"threshold={info.threshold:.3f}, stride={info.stride}"
    )
