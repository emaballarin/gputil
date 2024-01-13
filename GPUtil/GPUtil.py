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
import math
import os
import platform
import random
import subprocess
import sys
import time
from typing import List

from distutils import spawn


__version__ = "1.5.3"


class GPU:
    __slots__ = [
        "id",
        "uuid",
        "load",
        "memoryUtil",
        "memoryTotal",
        "memoryUsed",
        "memoryFree",
        "driver",
        "name",
        "serial",
        "display_mode",
        "display_active",
        "temperature",
        "vbios_version",
        "power_draw",
        "power_limit",
        "core_clock",
        "memory_clock",
        "compute_mode",
        "pci_bus",
    ]

    def __init__(
        self,
        ID,
        uuid,
        load,
        memoryTotal,
        memoryUsed,
        memoryFree,
        driver,
        gpu_name,
        serial,
        display_mode,
        display_active,
        temp_gpu,
        core_clock,
        memory_clock,
        vbios_version,
        power_draw,
        power_limit,
        compute_mode,
        pci_bus,
    ):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memoryUsed) / float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu
        self.vbios_version = vbios_version
        self.power_draw = power_draw
        self.power_limit = power_limit
        self.core_clock = core_clock
        self.memory_clock = memory_clock
        self.compute_mode = compute_mode
        self.pci_bus = pci_bus

    def __str__(self):
        return str(self.__dict__)


class GPUProcess:
    __slots__ = [
        "pid",
        "processName",
        "gpuId",
        "gpuUuid",
        "gpuName",
        "usedMemory",
        "uid",
        "uname",
    ]

    def __init__(
        self, pid, processName, gpuId, gpuUuid, gpuName, usedMemory, uid, uname
    ):
        self.pid = pid
        self.processName = processName
        self.gpuId = gpuId
        self.gpuUuid = gpuUuid
        self.gpuName = gpuName
        self.usedMemory = usedMemory
        self.uid = uid
        self.uname = uname

    def __str__(self):
        return str(self.__dict__)


def safeFloatCast(strNumber: str) -> float:
    try:
        number = float(strNumber)
    except ValueError:
        number = float("nan")
    return number


def getNvidiaSmiCmd() -> str:
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi
        # could not be found from the environment path,
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable("nvidia-smi")
        if nvidia_smi is None:
            nvidia_smi = (
                "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
                % os.environ["systemdrive"]
            )
    else:
        nvidia_smi = "nvidia-smi"
    return nvidia_smi


def getGPUs() -> List[GPU]:
    # Get ID, processing and memory utilization for all GPUs
    nvidia_smi = getNvidiaSmiCmd()
    try:
        p = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu,clocks.current.graphics,clocks.current.memory,vbios_version,power.draw,power.limit,compute_mode,pci.bus",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        stdout, stderror = p.stdout, p.stderr
    except:
        return []
    output = stdout
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    numDevices = len(lines) - 1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        # print(line)
        vals = line.split(", ")
        # print(vals)
        deviceIds = int(vals[0])
        uuid = vals[1]
        gpuUtil = safeFloatCast(vals[2]) / 100
        memTotal = safeFloatCast(vals[3])
        memUsed = safeFloatCast(vals[4])
        memFree = safeFloatCast(vals[5])
        driver = vals[6]
        gpu_name = vals[7]
        serial = vals[8]
        display_active = vals[9]
        display_mode = vals[10]
        temp_gpu = safeFloatCast(vals[11])
        core_clock = int(vals[12])
        memory_clock = int(vals[13])
        vbios_version = vals[14]
        power_draw = safeFloatCast(vals[15])
        power_limit = safeFloatCast(vals[16])
        compute_mode = vals[17]
        pci_bus = int(vals[18], 16)
        GPUs.append(
            GPU(
                deviceIds,
                uuid,
                gpuUtil,
                memTotal,
                memUsed,
                memFree,
                driver,
                gpu_name,
                serial,
                display_mode,
                display_active,
                temp_gpu,
                core_clock,
                memory_clock,
                vbios_version,
                power_draw,
                power_limit,
                compute_mode,
                pci_bus,
            )
        )
    return GPUs


def getGPUProcesses() -> List[GPUProcess]:
    """Get all gpu compute processes."""

    global gpuUuidToIdMap

    nvidia_smi = getNvidiaSmiCmd()
    try:
        p = subprocess.run(
            [
                nvidia_smi,
                "--query-compute-apps=pid,process_name,gpu_uuid,gpu_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            encoding="utf8",
        )
        stdout, _ = p.stdout, p.stderr
    except:
        return []
    output = stdout
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    numProcesses = len(lines) - 1
    processes = []
    for g in range(numProcesses):
        line = lines[g]
        # print(line)
        vals = line.split(", ")
        # print(vals)
        pid = int(vals[0])
        processName = vals[1]
        gpuUuid = vals[2]
        gpuName = vals[3]
        usedMemory = safeFloatCast(vals[4])
        gpuId = gpuUuidToIdMap[gpuUuid]
        if gpuId is None:
            gpuId = -1

        # get uid and uname owner of the pid
        try:
            p = subprocess.run(
                ["ps", f"-p{pid}", "-oruid=,ruser="],
                stdout=subprocess.PIPE,
                encoding="utf8",
            )
            uid, uname = p.stdout.split()
            uid = int(uid)
        except:
            uid, uname = -1, ""

        processes.append(
            GPUProcess(
                pid, processName, gpuId, gpuUuid, gpuName, usedMemory, uid, uname
            )
        )
    return processes


def getAvailable(
    order="first",
    limit=1,
    maxLoad=0.5,
    maxMemory=0.5,
    memoryFree=0,
    includeNan=False,
    excludeID=[],
    excludeUUID=[],
    excludeComputeMode=["Exclusive_Process"],
) -> List[int]:
    # order = first | last | random | load | memory
    #    first --> select the GPU with the lowest ID (DEFAULT)
    #    last --> select the GPU with the highest ID
    #    random --> select a random available GPU
    #    load --> select the GPU with the lowest load
    #    memory --> select the GPU with the most memory available
    # limit = 1 (DEFAULT), 2, ..., Inf
    #     Limit sets the upper limit for the number of GPUs to return. E.g. if limit = 2, but only one is available, only one is returned.

    # Get device IDs, load and memory usage
    GPUs = getGPUs()

    # Determine, which GPUs are available
    GPUavailability = getAvailability(
        GPUs,
        maxLoad=maxLoad,
        maxMemory=maxMemory,
        memoryFree=memoryFree,
        includeNan=includeNan,
        excludeID=excludeID,
        excludeUUID=excludeUUID,
        excludeComputeMode=excludeComputeMode,
    )
    availAbleGPUindex = [
        idx for idx in range(0, len(GPUavailability)) if (GPUavailability[idx] == 1)
    ]
    # Discard unavailable GPUs
    GPUs = [GPUs[g] for g in availAbleGPUindex]

    # Sort available GPUs according to the order argument
    if order == "first":
        GPUs.sort(
            key=lambda x: float("inf") if math.isnan(x.id) else x.id, reverse=False
        )
    elif order == "last":
        GPUs.sort(
            key=lambda x: float("-inf") if math.isnan(x.id) else x.id, reverse=True
        )
    elif order == "random":
        GPUs = [GPUs[g] for g in random.sample(range(0, len(GPUs)), len(GPUs))]
    elif order == "load":
        GPUs.sort(
            key=lambda x: float("inf") if math.isnan(x.load) else x.load, reverse=False
        )
    elif order == "memory":
        GPUs.sort(
            key=lambda x: float("inf") if math.isnan(x.memoryUtil) else x.memoryUtil,
            reverse=False,
        )

    # Extract the number of desired GPUs, but limited to the total number of available GPUs
    GPUs = GPUs[0 : min(limit, len(GPUs))]

    # Extract the device IDs from the GPUs and return them
    deviceIds = [gpu.id for gpu in GPUs]

    return deviceIds


def getAvailability(
    GPUs,
    maxLoad=0.5,
    maxMemory=0.5,
    memoryFree=0,
    includeNan=False,
    excludeID=[],
    excludeUUID=[],
    excludeComputeMode=["Exclusive_Process"],
) -> List[int]:
    # Determine, which GPUs are available
    GPUavailability = [
        1
        if (gpu.memoryFree >= memoryFree)
        and (gpu.load < maxLoad or (includeNan and math.isnan(gpu.load)))
        and (gpu.memoryUtil < maxMemory or (includeNan and math.isnan(gpu.memoryUtil)))
        and (
            (gpu.id not in excludeID)
            and (gpu.uuid not in excludeUUID)
            and (gpu.compute_mode not in excludeComputeMode)
        )
        else 0
        for gpu in GPUs
    ]
    return GPUavailability


def getFirstAvailable(
    order="first",
    maxLoad=0.5,
    maxMemory=0.5,
    attempts=1,
    interval=900,
    verbose=False,
    includeNan=False,
    excludeID=[],
    excludeUUID=[],
    excludeComputeMode=["Exclusive_Process"],
):
    for i in range(attempts):
        if verbose:
            print(
                "Attempting ("
                + str(i + 1)
                + "/"
                + str(attempts)
                + ") to locate available GPU."
            )
        # Get first available GPU
        available = getAvailable(
            order=order,
            limit=1,
            maxLoad=maxLoad,
            maxMemory=maxMemory,
            includeNan=includeNan,
            excludeID=excludeID,
            excludeUUID=excludeUUID,
            excludeComputeMode=excludeComputeMode,
        )
        # If an available GPU was found, break for loop.
        if available:
            if verbose:
                print("GPU " + str(available) + " located!")
            break
        # If this is not the last attempt, sleep for 'interval' seconds
        if i != attempts - 1:
            time.sleep(interval)
    # Check if an GPU was found, or if the attempts simply ran out. Throw error, if no GPU was found
    if not available:
        raise RuntimeError(
            "Could not find an available GPU after "
            + str(attempts)
            + " attempts with "
            + str(interval)
            + " seconds interval."
        )

    # Return found GPU
    return available


def showUtilization(all=False, attrList=None, useOldCode=False):
    GPUs = getGPUs()
    if all:
        if useOldCode:
            print(
                " ID | Name | Serial | UUID || GPU util. | Memory util. || Memory total | Memory used | Memory free || Display mode | Display active || Core Clock | Memory Clock || Power draw || Compute Mode || PCI Id"
            )
            print(
                "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
            )
            for gpu in GPUs:
                print(
                    " {0:2d} | {1:s} | {2:s} | {3:s} || {4:3.0f}% | {5:3.0f}% || {6:.0f}MB | {7:.0f}MB | {8:.0f}MB || {9:s} | {10:s} || {11:d}Mhz | {12:d}Mhz || {13:2.0f}W || {14:s} || {15:X}".format(
                        gpu.id,
                        gpu.name,
                        gpu.serial,
                        gpu.uuid,
                        gpu.load * 100,
                        gpu.memoryUtil * 100,
                        gpu.memoryTotal,
                        gpu.memoryUsed,
                        gpu.memoryFree,
                        gpu.display_mode,
                        gpu.display_active,
                        gpu.core_clock,
                        gpu.memory_clock,
                        gpu.power_draw,
                        gpu.compute_mode,
                        gpu.pci_bus,
                    )
                )
        else:
            attrList = [
                [
                    {"attr": "id", "name": "ID"},
                    {"attr": "name", "name": "Name"},
                    {"attr": "serial", "name": "Serial"},
                    {"attr": "uuid", "name": "UUID"},
                ],
                [
                    {
                        "attr": "temperature",
                        "name": "GPU temp.",
                        "suffix": "C",
                        "transform": lambda x: x,
                        "precision": 0,
                    },
                    {
                        "attr": "load",
                        "name": "GPU util.",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                    {
                        "attr": "memoryUtil",
                        "name": "Memory util.",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                ],
                [
                    {
                        "attr": "memoryTotal",
                        "name": "Memory total",
                        "suffix": "MB",
                        "precision": 0,
                    },
                    {
                        "attr": "memoryUsed",
                        "name": "Memory used",
                        "suffix": "MB",
                        "precision": 0,
                    },
                    {
                        "attr": "memoryFree",
                        "name": "Memory free",
                        "suffix": "MB",
                        "precision": 0,
                    },
                ],
                [
                    {"attr": "display_mode", "name": "Display mode"},
                    {"attr": "display_active", "name": "Display active"},
                ],
                [
                    {"attr": "core_clock", "name": "Core Clock"},
                    {"attr": "memory_clock", "name": "Memory Clock"},
                ],
                [
                    {"attr": "power_draw", "name": "Power draw"},
                ],
                [{"attr": "compute_mode", "name": "Compute mode"}],
                [{"attr": "pci_bus", "name": "PCI bus ID"}],
            ]

    else:
        if useOldCode:
            print(" ID  GPU  MEM")
            print("--------------")
            for gpu in GPUs:
                print(
                    " {0:2d} {1:3.0f}% {2:3.0f}%".format(
                        gpu.id, gpu.load * 100, gpu.memoryUtil * 100
                    )
                )
        else:
            # if `attrList` was not specified, use the default one
            attrList = [
                [
                    {"attr": "id", "name": "ID"},
                    {
                        "attr": "load",
                        "name": "GPU",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                    {
                        "attr": "memoryUtil",
                        "name": "MEM",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                ],
            ]

    if not useOldCode and attrList is not None:
        headerString = ""
        GPUstrings = [""] * len(GPUs)
        for attrGroup in attrList:
            # print(attrGroup)
            for attrDict in attrGroup:
                headerString = headerString + "| " + attrDict["name"] + " "
                headerWidth = len(attrDict["name"])
                minWidth = len(attrDict["name"])

                attrPrecision = (
                    "." + str(attrDict["precision"])
                    if ("precision" in attrDict.keys())
                    else ""
                )
                attrSuffix = (
                    str(attrDict["suffix"]) if ("suffix" in attrDict.keys()) else ""
                )
                attrTransform = (
                    attrDict["transform"]
                    if ("transform" in attrDict.keys())
                    else lambda x: x
                )
                for gpu in GPUs:
                    attr = getattr(gpu, attrDict["attr"])

                    attr = attrTransform(attr)

                    if isinstance(attr, float):
                        attrStr = ("{0:" + attrPrecision + "f}").format(attr)
                    elif isinstance(attr, int):
                        attrStr = "{0:d}".format(attr)
                    elif isinstance(attr, str):
                        attrStr = attr
                    elif sys.version_info[0] == 2:
                        if isinstance(attr, str):
                            attrStr = attr.encode("ascii", "ignore")
                    else:
                        raise TypeError(
                            "Unhandled object type ("
                            + str(type(attr))
                            + ") for attribute '"
                            + attrDict["name"]
                            + "'"
                        )

                    attrStr += attrSuffix

                    minWidth = max(minWidth, len(attrStr))

                headerString += " " * max(0, minWidth - headerWidth)

                minWidthStr = str(minWidth - len(attrSuffix))

                for gpuIdx, gpu in enumerate(GPUs):
                    attr = getattr(gpu, attrDict["attr"])

                    attr = attrTransform(attr)

                    if isinstance(attr, float):
                        attrStr = ("{0:" + minWidthStr + attrPrecision + "f}").format(
                            attr
                        )
                    elif isinstance(attr, int):
                        attrStr = ("{0:" + minWidthStr + "d}").format(attr)
                    elif isinstance(attr, str):
                        attrStr = ("{0:" + minWidthStr + "s}").format(attr)
                    elif sys.version_info[0] == 2:
                        if isinstance(attr, str):
                            attrStr = ("{0:" + minWidthStr + "s}").format(
                                attr.encode("ascii", "ignore")
                            )
                    else:
                        raise TypeError(
                            "Unhandled object type ("
                            + str(type(attr))
                            + ") for attribute '"
                            + attrDict["name"]
                            + "'"
                        )

                    attrStr += attrSuffix

                    GPUstrings[gpuIdx] += "| " + attrStr + " "

            headerString = headerString + "|"
            for gpuIdx, gpu in enumerate(GPUs):
                GPUstrings[gpuIdx] += "|"

        headerSpacingString = "-" * len(headerString)
        print(headerString)
        print(headerSpacingString)
        for GPUstring in GPUstrings:
            print(GPUstring)


# Generate gpu uuid to id map
gpuUuidToIdMap = {}
try:
    gpus = getGPUs()
    for gpu in gpus:
        gpuUuidToIdMap[gpu.uuid] = gpu.id
    del gpus
except:
    pass
