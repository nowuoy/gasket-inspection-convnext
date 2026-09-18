from ctypes import *
import sys
import os
import time
import queue
import threading
import traceback

import numpy as np
import cv2

sys.path.insert(0, "/opt/MVS/Samples/64/Python")

from MvImport.MvCameraControl_class import *


# ============================================================
# 사용자 설정
# ============================================================

CAMERAS = {
    "cam1": "DA7552809",
    "cam2": "DA7552858",
    "cam3": "DA7538923",
    "cam4": "DA7538915",
}

STATUS_LABEL = "test" ##############

SAVE_ROOT = os.path.expanduser(
    os.path.join("/home/anchor/Documents/Codex/신경망 생성 이어서/gasket-inspection-convnext/runtime/inbox")
)
TIMEOUT_MS = 1000

# 카메라 내부에서 대기시킬 수신 프레임 개수
IMAGE_NODE_COUNT = 20

# PNG 저장 대기 큐 크기
SAVE_QUEUE_SIZE = 40

# GigE 카메라 전송 시점 분산값
#
# 카메라 모델에 따라 단위가 다를 수 있습니다.
# 처음에는 아래 값으로 시험하고 문제가 계속되면 간격을 늘립니다.
#
# 예:
# 0, 100000, 200000, 300000
#
TRANSMISSION_DELAYS = {
    "cam1": 0,
    "cam2": 100000,
    "cam3": 200000,
    "cam4": 300000,
}

# 패킷 사이 간격
#
# 4대가 하나의 NIC 또는 스위치를 공유한다면 값을 늘리는 것이
# 패킷 유실 방지에 도움이 될 수 있습니다.
INTER_PACKET_DELAY = 5000


stop_event = threading.Event()

save_queue = queue.Queue(maxsize=SAVE_QUEUE_SIZE)

capture_counts = {
    "cam1": 0,
    "cam2": 0,
    "cam3": 0,
    "cam4": 0,
}

count_lock = threading.Lock()


# ============================================================
# SDK 오류 코드 표시
# ============================================================

def hex_ret(ret):
    return f"0x{ret & 0xFFFFFFFF:08X}"


# ============================================================
# Serial Number 읽기
# ============================================================

def get_serial(device_info):
    if device_info.nTLayerType == MV_GIGE_DEVICE:
        info = device_info.SpecialInfo.stGigEInfo

    elif device_info.nTLayerType == MV_USB_DEVICE:
        info = device_info.SpecialInfo.stUsb3VInfo

    else:
        return None

    return bytes(info.chSerialNumber).split(b"\0")[0].decode(
        "ascii",
        errors="ignore",
    )


# ============================================================
# 카메라별 촬영 번호
# ============================================================

def get_next_object_number(camera_name):
    with count_lock:
        capture_counts[camera_name] += 1
        return capture_counts[camera_name]


# ============================================================
# 카메라 Feature 설정 보조 함수
# ============================================================

def try_set_int(cam, camera_name, feature_name, value):
    ret = cam.MV_CC_SetIntValue(feature_name, int(value))

    if ret == 0:
        print(
            f"[{camera_name}] {feature_name} = {value}"
        )
        return True

    print(
        f"[{camera_name}] {feature_name} 설정 건너뜀: "
        f"{hex_ret(ret)}"
    )
    return False


# ============================================================
# GigE 네트워크 설정
# ============================================================

def configure_gige_transport(
    cam,
    camera_name,
    device_info,
):
    if device_info.nTLayerType != MV_GIGE_DEVICE:
        print(
            f"[{camera_name}] GigE 카메라가 아니므로 "
            "네트워크 설정을 건너뜁니다."
        )
        return

    # 해당 NIC와 카메라 연결에서 사용할 수 있는 최적 패킷 크기
    packet_size = cam.MV_CC_GetOptimalPacketSize()

    if packet_size > 0:
        ret = cam.MV_CC_SetIntValue(
            "GevSCPSPacketSize",
            packet_size,
        )

        if ret == 0:
            print(
                f"[{camera_name}] optimal packet size = "
                f"{packet_size}"
            )
        else:
            print(
                f"[{camera_name}] packet size 설정 실패: "
                f"{hex_ret(ret)}"
            )
    else:
        print(
            f"[{camera_name}] optimal packet size 조회 실패: "
            f"{packet_size}"
        )

    # 패킷 사이 간격
    try_set_int(
        cam,
        camera_name,
        "GevSCPD",
        INTER_PACKET_DELAY,
    )

    # 카메라별 프레임 전송 시작 시점을 분산
    #
    # 카메라 펌웨어에 따라 GevSCFTD가 없거나 쓰기 불가능할 수 있습니다.
    # 그 경우 오류를 출력하고 계속 실행합니다.
    try_set_int(
        cam,
        camera_name,
        "GevSCFTD",
        TRANSMISSION_DELAYS[camera_name],
    )


# ============================================================
# PNG 저장 전용 Worker
# ============================================================

def save_worker():
    print("[SAVE] worker started")

    while not stop_event.is_set() or not save_queue.empty():
        try:
            item = save_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            camera_name = item["camera_name"]
            object_number = item["object_number"]
            frame_number = item["frame_number"]
            image = item["image"]

            camera_number = camera_name.replace("cam", "")

            filename = os.path.join(
                SAVE_ROOT,
                (
                    f"{STATUS_LABEL}_"
                    f"object_{object_number:04d}_"
                    f"camera_{camera_number}.png"
                ),
            )

            success = cv2.imwrite(filename, image)

            if success:
                print(
                    f"[{camera_name}] saved "
                    f"object={object_number:04d}, "
                    f"frame={frame_number}: "
                    f"{filename}"
                )
            else:
                print(
                    f"[{camera_name}] SAVE FAILED: "
                    f"{filename}"
                )

        except Exception:
            print("[SAVE] worker exception")
            traceback.print_exc()

        finally:
            save_queue.task_done()

    print("[SAVE] worker stopped")


# ============================================================
# Camera Worker
# ============================================================

def camera_worker(camera_name, cam, payload_size):
    print(
        f"[{camera_name}] worker started, "
        f"payload={payload_size} bytes"
    )

    # BGR 변환 시 Mono/Bayer 원본보다 최대 3배가 필요할 수 있습니다.
    #
    # payload_size만 사용하면 Bayer/Mono 원본 크기로 계산될 수 있으므로
    # 안전하게 3배 이상 확보합니다.
    buffer_size = max(
        payload_size * 3,
        1024 * 1024,
    )

    data_buffer = (c_ubyte * buffer_size)()
    frame_info = MV_FRAME_OUT_INFO_EX()

    last_status_print = time.monotonic()
    previous_frame_number = None

    try:
        while not stop_event.is_set():
            memset(
                byref(frame_info),
                0,
                sizeof(frame_info),
            )

            ret = cam.MV_CC_GetImageForBGR(
                data_buffer,
                buffer_size,
                frame_info,
                TIMEOUT_MS,
            )

            if ret != 0:
                now = time.monotonic()

                if now - last_status_print >= 5.0:
                    print(
                        f"[{camera_name}] waiting / "
                        f"SDK ret={hex_ret(ret)} / "
                        f"save_queue={save_queue.qsize()}"
                    )
                    last_status_print = now

                continue

            width = int(frame_info.nWidth)
            height = int(frame_info.nHeight)
            frame_number = int(frame_info.nFrameNum)
            required_size = width * height * 3

            # 비정상 메타데이터 또는 버퍼 부족 검출
            if width <= 0 or height <= 0:
                print(
                    f"[{camera_name}] invalid frame size: "
                    f"{width}x{height}"
                )
                continue

            if required_size > buffer_size:
                print(
                    f"[{camera_name}] BGR buffer too small: "
                    f"required={required_size}, "
                    f"allocated={buffer_size}"
                )
                continue

            # 프레임 번호 누락 감시
            if previous_frame_number is not None:
                expected = previous_frame_number + 1

                if frame_number != expected:
                    print(
                        f"[{camera_name}] WARNING: "
                        f"frame number jump "
                        f"{previous_frame_number} -> "
                        f"{frame_number}"
                    )

            previous_frame_number = frame_number

            # ctypes 버퍼를 NumPy 배열로 해석
            image_view = np.ctypeslib.as_array(
                data_buffer,
                shape=(buffer_size,),
            )

            # 중요:
            # 다음 GetImageForBGR 호출에서 data_buffer가 덮어써지므로
            # 저장 큐에 넣기 전에 반드시 copy() 해야 합니다.
            image = (
                image_view[:required_size]
                .reshape(height, width, 3)
                .copy()
            )

            object_number = get_next_object_number(
                camera_name
            )

            save_item = {
                "camera_name": camera_name,
                "object_number": object_number,
                "frame_number": frame_number,
                "image": image,
            }

            try:
                # 큐가 가득 차면 이미지를 조용히 버리지 않고
                # 최대 2초 동안 저장 Worker를 기다립니다.
                save_queue.put(save_item, timeout=2.0)

            except queue.Full:
                print(
                    f"[{camera_name}] ERROR: save queue full. "
                    f"Frame {frame_number} was not saved. "
                    "저장 속도보다 촬영 속도가 빠릅니다."
                )

            print(
                f"[{camera_name}] FRAME RECEIVED "
                f"object={object_number:04d}, "
                f"frame={frame_number}, "
                f"size={width}x{height}, "
                f"queue={save_queue.qsize()}"
            )

            last_status_print = time.monotonic()

    except Exception:
        print(f"[{camera_name}] WORKER CRASHED")
        traceback.print_exc()

    finally:
        print(f"[{camera_name}] worker stopped")


# ============================================================
# Main
# ============================================================

def main():
    stop_event.clear()

    os.makedirs(SAVE_ROOT, exist_ok=True)

    device_list = MV_CC_DEVICE_INFO_LIST()

    ret = MvCamera.MV_CC_EnumDevices(
        MV_GIGE_DEVICE | MV_USB_DEVICE,
        device_list,
    )

    if ret != 0:
        print(
            f"Camera enumeration failed: "
            f"{hex_ret(ret)}"
        )
        return

    print(f"Camera found: {device_list.nDeviceNum}")

    found_devices = {}

    for i in range(device_list.nDeviceNum):
        device_pointer = cast(
            device_list.pDeviceInfo[i],
            POINTER(MV_CC_DEVICE_INFO),
        )

        # contents 객체만 보관하지 않고 구조체를 복사합니다.
        # device_list의 수명과 분리하기 위한 방어적인 처리입니다.
        device_info = MV_CC_DEVICE_INFO()

        memmove(
            byref(device_info),
            device_pointer,
            sizeof(MV_CC_DEVICE_INFO),
        )

        serial = get_serial(device_info)

        if serial:
            found_devices[serial] = device_info
            print(f"Camera {i}: {serial}")

    for name, serial in CAMERAS.items():
        if serial not in found_devices:
            print(f"{name} not found: {serial}")
            return

    opened_cameras = {}
    camera_threads = []
    save_thread = None

    try:
        # ----------------------------------------------------
        # Open 및 네트워크 설정
        # ----------------------------------------------------

        for name, serial in CAMERAS.items():
            cam = MvCamera()
            device_info = found_devices[serial]

            ret = cam.MV_CC_CreateHandle(device_info)

            if ret != 0:
                raise RuntimeError(
                    f"{name} CreateHandle failed: "
                    f"{hex_ret(ret)}"
                )

            ret = cam.MV_CC_OpenDevice(
                MV_ACCESS_Exclusive,
                0,
            )

            if ret != 0:
                try:
                    cam.MV_CC_DestroyHandle()
                except Exception:
                    pass

                raise RuntimeError(
                    f"{name} OpenDevice failed: "
                    f"{hex_ret(ret)}"
                )

            opened_cameras[name] = {
                "cam": cam,
                "device_info": device_info,
            }

            print(f"{name} open success")

            # SDK 내부 이미지 노드 수 증가
            ret = cam.MV_CC_SetImageNodeNum(
                IMAGE_NODE_COUNT
            )

            if ret != 0:
                print(
                    f"[{name}] SetImageNodeNum failed: "
                    f"{hex_ret(ret)}"
                )

            configure_gige_transport(
                cam,
                name,
                device_info,
            )

        # ----------------------------------------------------
        # Payload 크기 조회
        # ----------------------------------------------------

        for name, camera_data in opened_cameras.items():
            cam = camera_data["cam"]

            payload_value = MVCC_INTVALUE()
            memset(
                byref(payload_value),
                0,
                sizeof(payload_value),
            )

            ret = cam.MV_CC_GetIntValue(
                "PayloadSize",
                payload_value,
            )

            if ret != 0:
                raise RuntimeError(
                    f"{name} PayloadSize failed: "
                    f"{hex_ret(ret)}"
                )

            camera_data["payload_size"] = int(
                payload_value.nCurValue
            )

            print(
                f"[{name}] PayloadSize = "
                f"{camera_data['payload_size']}"
            )

        # ----------------------------------------------------
        # Start grabbing
        # ----------------------------------------------------

        for name, camera_data in opened_cameras.items():
            cam = camera_data["cam"]

            ret = cam.MV_CC_StartGrabbing()

            if ret != 0:
                raise RuntimeError(
                    f"{name} StartGrabbing failed: "
                    f"{hex_ret(ret)}"
                )

            print(f"{name} grabbing started")

        # ----------------------------------------------------
        # PNG 저장 Worker 시작
        # ----------------------------------------------------

        save_thread = threading.Thread(
            target=save_worker,
            name="png-save-worker",
            daemon=False,
        )
        save_thread.start()

        # ----------------------------------------------------
        # 카메라별 수신 Worker 시작
        # ----------------------------------------------------

        for name, camera_data in opened_cameras.items():
            thread = threading.Thread(
                target=camera_worker,
                args=(
                    name,
                    camera_data["cam"],
                    camera_data["payload_size"],
                ),
                name=f"{name}-worker",
                daemon=False,
            )

            thread.start()
            camera_threads.append(thread)

        print()
        print("====================================")
        print("ALL CAMERAS READY")
        print("Press Ctrl+C to stop.")
        print("====================================")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    except Exception:
        print("\nMAIN ERROR")
        traceback.print_exc()

    finally:
        stop_event.set()

        # GetImageForBGR timeout보다 충분히 길게 기다립니다.
        for thread in camera_threads:
            thread.join(timeout=(TIMEOUT_MS / 1000.0) + 2.0)

        # 이미 수신한 프레임 저장 완료 대기
        if save_thread is not None:
            print(
                f"Waiting for {save_queue.qsize()} "
                "queued images..."
            )
            save_queue.join()
            save_thread.join(timeout=5.0)

        for name, camera_data in opened_cameras.items():
            cam = camera_data["cam"]

            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass

            try:
                cam.MV_CC_CloseDevice()
            except Exception:
                pass

            try:
                cam.MV_CC_DestroyHandle()
            except Exception:
                pass

            print(f"{name} closed")


if __name__ == "__main__":
    main()
