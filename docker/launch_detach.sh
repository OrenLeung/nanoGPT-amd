#!/usr/bin/env bash


sudo docker run -d --privileged --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 192G \
    -v .:/workspace/llm-train-bench/ \
    --name llm_persistent_container \
    llm-train-bench \
    tail -f /dev/null
