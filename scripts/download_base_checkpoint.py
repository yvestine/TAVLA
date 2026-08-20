"""Download the public pi0_base checkpoint and assets into the OpenPI cache."""

from __future__ import annotations

import argparse

from openpi.shared.download import maybe_download


BASE_URL = "s3://openpi-assets/checkpoints/pi0_base"


def main() -> None:
    checkpoint = maybe_download(f"{BASE_URL}/params")
    assets = maybe_download(f"{BASE_URL}/assets")
    print(f"pi0_base params: {checkpoint}")
    print(f"pi0_base assets: {assets}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
