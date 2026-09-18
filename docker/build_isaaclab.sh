#!/usr/bin/env bash
# Build Isaac Lab's own image (its docker/Dockerfile.base) at a pinned commit of the
# release/3.0.0 branch: Isaac Sim 6.1.0, Isaac Lab 3.0, rsl-rl-lib 5.5.1. docker/Dockerfile.sim
# builds on the result. Needed once per pin; slow.
#
# The Dockerfile is patched on the fly so uv downloads fewer wheels at once: with uv's default
# of 50, the multi-hundred-MB CUDA wheels from pypi.nvidia.com time out on a ~20 Mbit/s link.
# Set UV_CONCURRENT_DOWNLOADS to change it.
#
# To move to a newer Isaac Lab, change ISAACLAB_COMMIT here and ISAACLAB_IMAGE in
# Dockerfile.sim.
set -euo pipefail

ISAACLAB_COMMIT=11508e5c80a98e215b20cca842995d3836ff308d
TAG="gaitnet/isaaclab:3.0.0-${ISAACLAB_COMMIT:0:7}"
: "${UV_CONCURRENT_DOWNLOADS:=4}"

dockerfile=$(curl -fsSL "https://raw.githubusercontent.com/isaac-sim/IsaacLab/${ISAACLAB_COMMIT}/docker/Dockerfile.base")
patched=$(sed "/^ENV UV_HTTP_RETRIES=/a ENV UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS}\nENV UV_HTTP_TIMEOUT=300" <<<"$dockerfile")
if ! grep -q "^ENV UV_CONCURRENT_DOWNLOADS=" <<<"$patched"; then
    echo "Isaac Lab's Dockerfile.base has no 'ENV UV_HTTP_RETRIES=' line to patch after" >&2
    exit 1
fi

# only the extras we use; Isaac Lab's default image also carries sb3, skrl, rl-games, ...
docker buildx build --load -t "$TAG" -f - \
    --build-arg ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim \
    --build-arg ISAACSIM_VERSION_ARG=6.1.0 \
    --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
    --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab \
    --build-arg DOCKER_USER_HOME_ARG=/root \
    --build-arg IMAGE_EXTRAS="--extra rsl-rl --extra test" \
    "$@" \
    "https://github.com/isaac-sim/IsaacLab.git#${ISAACLAB_COMMIT}" <<<"$patched"
echo "built $TAG"
