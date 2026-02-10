import numpy as np
from pathlib import Path
from PIL import Image

TARGET_DIR = Path(__file__).parent / "src" / "aiperf" / "dataset" / "generator" / "assets" / "source_images"
NUM_IMAGES = 1000
WIDTH = 512
HEIGHT = 512


def main() -> None:
    rng = np.random.default_rng(42)
    for i in range(NUM_IMAGES):
        pixels = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(TARGET_DIR / f"noise_{i:04d}.png")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{NUM_IMAGES}")


if __name__ == "__main__":
    main()
