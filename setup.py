# /usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# GPUtil - GPU utilization
#
# A Python module for programmically getting the GPU utilization from NVIDIA
# GPUs using `nvidia-smi`.
#
# Copyright (c) 2017-2019 Anders Krogh Mortensen (anderskm)
#                         [https://github.com/anderskm/gputil]
#                    2019 Zhou Shengsheng        (ZhouShengsheng)
#                    2020 Daniel Laguna          (labellson)
#                    2020 Elliott Balsley        (llamafilm)
#                    2020 Max Lv                 (madeye)
#                    2022 Min Sheng Wu 'Vincent' (Min-Sheng)
#                    2022 Romanin                (romanin-rf)
#                    2023 Aaron Sheldon          (aaronsheldon)
#                    2023 Cristian Brotto        (brottobhmg)
#                    2024 Emanuele Ballarin      (emaballarin)
#                         [https://github.com/emaballarin/gputil]
#
# Licensed under the MIT License.
# ──────────────────────────────────────────────────────────────────────────────
from distutils.core import setup

import GPUtil

setup(
    name="GPUtil",
    packages=["GPUtil"],
    version=GPUtil.__version__,
    description="GPUtil is a Python module for getting the GPU status from NVIDIA GPUs using `nvidia-smi`.",
    author="Emanuele Ballarin",
    author_email="emanuele@ballarin.cc",
    url="https://github.com/emaballarin/gputil",
    download_url="https://fury.ballarin.cc/pypi/gputil",
    keywords=[
        "gpu",
        "utilization",
        "load",
        "memory",
        "available",
        "usage",
        "free",
        "select",
        "nvidia",
    ],
    classifiers=[],
    license="MIT",
)
