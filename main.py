# -*- coding: utf-8 -*-
"""
RTSP/UDP Viewer (GStreamer + OpenCV) — Windows/Linux

EXE-режим:
- PyInstaller --onedir
- рядом с exe кладём папку gst\
  gst\bin\                     (dll)
  gst\lib\girepository-1.0      (typelibs)
  gst\lib\gstreamer-1.0         (plugins)

При запуске из exe автоматически выставляет:
- PATH += gst\bin
- GI_TYPELIB_PATH = gst\lib\girepository-1.0
- GST_PLUGIN_PATH = gst\lib\gstreamer-1.0
- GST_REGISTRY_1_0 = %LOCALAPPDATA%\dvp_gst_registry.bin
"""

import os
import sys
import time
import threading
import numpy as np

# ----------------- CONFIG ПО УМОЛЧАНИЮ -----------------
DEFAULT_CONFIG = {
    "SOURCE_TYPE": "RTSP",                          # "UDP" или "RTSP"
    "UDP_PORT": 5600,
    "RTSP_URL": "rtsp://root:12345@192.168.1.24/stream=0",
    "PAYLOAD": 97,
    "CODEC": "H264",                                # "H264" или "H265"
    "BUFFER_SIZE": 131072,
    "JITTER_LATENCY_MS": 0,
    "USE_OPENCV": True,
    "CANNY_T1": 150,
    "CANNY_T2": 170,
    "SCALE_PERCENT": 111,                           # масштаб окна вывода (в %)
    "SHOW_EDGES": True,                             # показывать контуры (окно count)
}

# ----------------- Runtime bootstrap (важно для exe) -----------------
def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def bootstrap_gst_runtime_from_local_folder():
    """
    Если рядом есть папка gst\, выставляем переменные окружения под неё,
    чтобы gi/GStreamer работали из распакованного приложения.
    """
    base = app_dir()
    gst_root = os.path.join(base, "gst")
    gst_bin = os.path.join(gst_root, "bin")
    gst_typelib = os.path.join(gst_root, "lib", "girepository-1.0")
    gst_plugins = os.path.join(gst_root, "lib", "gstreamer-1.0")

    if not os.path.exists(os.path.join(gst_bin, "gstreamer-1.0-0.dll")):
        return  # нет локального runtime

    if sys.platform.startswith("win"):
        try:
            os.add_dll_directory(gst_bin)
        except Exception:
            pass

    os.environ["PATH"] = gst_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ["GI_TYPELIB_PATH"] = gst_typelib

    os.environ["GST_PLUGIN_PATH"] = gst_plugins
    os.environ["GST_PLUGIN_SYSTEM_PATH_1_0"] = gst_plugins

    localapp = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    os.environ["GST_REGISTRY_1_0"] = os.path.join(localapp, "dvp_gst_registry.bin")

bootstrap_gst_runtime_from_local_folder()

# ----------------- GStreamer / GI -----------------
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GObject", "2.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GObject, GstApp
except Exception:
    print("Ошибка: требуется PyGObject (gi) + GStreamer Python bindings.", file=sys.stderr)
    print("Подсказка: запускай из conda env gstwin ИЛИ положи рядом папку gst\\ (runtime).", file=sys.stderr)
    raise

Gst.init(None)

# ----------------- Pipeline helpers -----------------
def pick_decoder(codec_str: str) -> str:
    codec = codec_str.upper()
    if codec == "H265":
        candidates = ["d3d11h265dec", "nvh265dec", "vaapih265dec", "avdec_h265"]
    elif codec == "H264":
        candidates = ["d3d11h264dec", "nvh264dec", "vaapih264dec", "avdec_h264"]
    else:
        raise ValueError("CODEC должен быть 'H265' или 'H264'")

    for name in candidates:
        if Gst.ElementFactory.find(name) is not None:
            return name
    return candidates[-1]

def build_pipeline_string(cfg: dict) -> str:
    codec = cfg["CODEC"].upper()
    depay = f"rtp{codec.lower()}depay"
    parse = f"{codec.lower()}parse"
    dec = pick_decoder(codec)

    source_type = cfg["SOURCE_TYPE"].upper()
    jitter = int(cfg["JITTER_LATENCY_MS"])
    use_opencv = bool(cfg["USE_OPENCV"])

    win_sink = "d3d11videosink" if Gst.ElementFactory.find("d3d11videosink") else "autovideosink"

    if source_type == "UDP":
        caps = (
            f'application/x-rtp,media=video,encoding-name={codec},'
            f'payload={cfg["PAYLOAD"]},clock-rate=90000'
        )
        if use_opencv:
            return (
                f'udpsrc port={cfg["UDP_PORT"]} buffer-size={cfg["BUFFER_SIZE"]} caps="{caps}" ! '
                f'rtpjitterbuffer latency={jitter} drop-on-latency=true ! '
                f'{depay} ! {parse} ! {dec} ! '
                f'queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! '
                f'videoconvert ! video/x-raw,format=BGR ! '
                f'appsink name=sink drop=true max-buffers=1 sync=false'
            )
        else:
            return (
                f'udpsrc port={cfg["UDP_PORT"]} buffer-size={cfg["BUFFER_SIZE"]} caps="{caps}" ! '
                f'rtpjitterbuffer latency={jitter} drop-on-latency=true ! '
                f'{depay} ! {parse} ! {dec} ! '
                f'queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! '
                f'videoconvert ! {win_sink} sync=false'
            )

    elif source_type == "RTSP":
        location = cfg["RTSP_URL"]
        if use_opencv:
            return (
                f'rtspsrc location="{location}" latency={jitter} protocols=tcp drop-on-latency=true ! '
                f'{depay} ! {parse} ! {dec} ! '
                f'queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! '
                f'videoconvert ! video/x-raw,format=BGR ! '
                f'appsink name=sink drop=true max-buffers=1 sync=false'
            )
        else:
            return (
                f'rtspsrc location="{location}" latency={jitter} protocols=tcp drop-on-latency=true ! '
                f'{depay} ! {parse} ! {dec} ! '
                f'queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! '
                f'videoconvert ! {win_sink} sync=false'
            )
    else:
        raise ValueError("SOURCE_TYPE должен быть 'UDP' или 'RTSP'")

# ----------------- Stream runner -----------------
def run_stream(cfg: dict, stop_event: threading.Event, status_cb=None):
    def set_status(msg: str):
        print(msg)
        if status_cb:
            status_cb(msg)

    use_opencv = bool(cfg.get("USE_OPENCV", True))
    try:
        import cv2
    except Exception:
        use_opencv = False
        set_status("[!] OpenCV недоступен, работаю без него")

    cfg = dict(cfg)
    cfg["USE_OPENCV"] = use_opencv

    # стартовые параметры
    t1 = int(cfg.get("CANNY_T1", 150))
    t2 = int(cfg.get("CANNY_T2", 170))
    show_edges = bool(cfg.get("SHOW_EDGES", True))
    scale_percent = int(cfg.get("SCALE_PERCENT", 100))
    scale_percent = max(10, min(400, scale_percent))

    try:
        pipeline_str = build_pipeline_string(cfg)
    except Exception as e:
        set_status(f"[ERR] Ошибка сборки пайплайна: {e}")
        return

    set_status("[i] GStreamer pipeline:")
    print(pipeline_str)

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        set_status(f"[ERR] Gst.parse_launch: {e}")
        return

    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)
    set_status("[i] PLAYING… (q/ESC выход, E контуры, +/- масштаб, 0 сброс)")

    # ---------- без OpenCV ----------
    if not use_opencv:
        while not stop_event.is_set():
            msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    set_status(f"[ERR] {err}; {debug}")
                    break
                if msg.type == Gst.MessageType.EOS:
                    set_status("[i] EOS")
                    break
        pipeline.set_state(Gst.State.NULL)
        set_status("[i] Пайплайн остановлен")
        return

    # ---------- OpenCV appsink ----------
    import cv2

    sink = pipeline.get_by_name("sink")
    if sink is None:
        set_status("[!] appsink не найден, не могу использовать OpenCV")
        pipeline.set_state(Gst.State.NULL)
        return

    # окна — делаем ресайзабельными
    cv2.namedWindow("normal", cv2.WINDOW_NORMAL)
    if show_edges:
        cv2.namedWindow("count", cv2.WINDOW_NORMAL)

    last_t = time.time()
    fps_acc = 0

    set_status(f"[i] Canny: {t1}, {t2} | edges={'ON' if show_edges else 'OFF'} | scale={scale_percent}%")

    def apply_scale(img_bgr):
        if scale_percent == 100:
            return img_bgr
        fx = scale_percent / 100.0
        fy = fx
        # для увеличения лучше linear, для уменьшения лучше area
        interp = cv2.INTER_LINEAR if fx >= 1.0 else cv2.INTER_AREA
        return cv2.resize(img_bgr, None, fx=fx, fy=fy, interpolation=interp)

    while not stop_event.is_set():
        sample = sink.emit("try-pull-sample", int(100 * Gst.MSECOND))
        if sample is None:
            msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    set_status(f"[ERR] {err}; {debug}")
                    break
                if msg.type == Gst.MessageType.EOS:
                    set_status("[i] EOS")
                    break
            continue

        buf = sample.get_buffer()
        caps = sample.get_caps()

        try:
            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")
        except Exception:
            width = height = None

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok or width is None or height is None:
            continue

        try:
            frame = np.ndarray((height, width, 3), buffer=mapinfo.data, dtype=np.uint8)

            # normal (масштабируем для показа)
            frame_show = apply_scale(frame)

            # небольшая подпись
            info = f"Scale {scale_percent}% | Edges {'ON' if show_edges else 'OFF'} | T1 {t1} T2 {t2}"
            cv2.putText(frame_show, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("normal", frame_show)

            if show_edges:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, threshold1=t1, threshold2=t2)
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                edges_show = apply_scale(edges_bgr)
                cv2.imshow("count", edges_show)

        finally:
            buf.unmap(mapinfo)

        fps_acc += 1
        now = time.time()
        if now - last_t >= 1.0:
            fps = fps_acc / (now - last_t)
            fps_acc = 0
            last_t = now
            set_status(f"[i] FPS ~ {fps:.1f} | edges={'ON' if show_edges else 'OFF'} | scale={scale_percent}%")

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            set_status("[i] Остановлено клавишей (q/ESC)")
            break

        # toggle edges
        if key in (ord("e"), ord("E")):
            show_edges = not show_edges
            if show_edges:
                cv2.namedWindow("count", cv2.WINDOW_NORMAL)
            else:
                try:
                    cv2.destroyWindow("count")
                except Exception:
                    pass
            set_status(f"[i] Edges {'ON' if show_edges else 'OFF'}")

        # scale +
        if key in (ord("+"), ord("=")):
            scale_percent = min(400, int(scale_percent * 1.1) if scale_percent >= 100 else scale_percent + 10)
            set_status(f"[i] Scale {scale_percent}%")

        # scale -
        if key in (ord("-"), ord("_")):
            scale_percent = max(10, int(scale_percent / 1.1) if scale_percent > 100 else scale_percent - 10)
            set_status(f"[i] Scale {scale_percent}%")

        # scale reset
        if key == ord("0"):
            scale_percent = 100
            set_status("[i] Scale 100%")

    pipeline.set_state(Gst.State.NULL)
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    set_status("[i] Пайплайн остановлен")

# ----------------- Tkinter GUI -----------------
import tkinter as tk
from tkinter import ttk

def create_gui():
    root = tk.Tk()
    root.title("RTSP/UDP Viewer (GStreamer + OpenCV)")
    root.geometry("560x380")

    status_var = tk.StringVar(value="Ожидаю запуск…")
    root.stop_event = threading.Event()
    root.viewer_thread = None

    source_type_var = tk.StringVar(value=DEFAULT_CONFIG["SOURCE_TYPE"])
    rtsp_url_var = tk.StringVar(value=DEFAULT_CONFIG["RTSP_URL"])
    udp_port_var = tk.StringVar(value=str(DEFAULT_CONFIG["UDP_PORT"]))
    payload_var = tk.StringVar(value=str(DEFAULT_CONFIG["PAYLOAD"]))
    codec_var = tk.StringVar(value=DEFAULT_CONFIG["CODEC"])
    jitter_var = tk.StringVar(value=str(DEFAULT_CONFIG["JITTER_LATENCY_MS"]))
    t1_var = tk.StringVar(value=str(DEFAULT_CONFIG["CANNY_T1"]))
    t2_var = tk.StringVar(value=str(DEFAULT_CONFIG["CANNY_T2"]))
    scale_var = tk.StringVar(value=str(DEFAULT_CONFIG["SCALE_PERCENT"]))
    edges_var = tk.BooleanVar(value=DEFAULT_CONFIG["SHOW_EDGES"])

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    row = 0
    ttk.Label(main_frame, text="Источник:").grid(row=row, column=0, sticky="w")
    ttk.Combobox(main_frame, textvariable=source_type_var, values=["RTSP", "UDP"], state="readonly", width=7)\
        .grid(row=row, column=1, sticky="w")
    row += 1

    ttk.Label(main_frame, text="RTSP URL:").grid(row=row, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=rtsp_url_var, width=50).grid(row=row, column=1, columnspan=3, sticky="we")
    row += 1

    ttk.Label(main_frame, text="UDP порт:").grid(row=row, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=udp_port_var, width=10).grid(row=row, column=1, sticky="w")
    row += 1

    ttk.Label(main_frame, text="RTP Payload:").grid(row=row, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=payload_var, width=10).grid(row=row, column=1, sticky="w")
    row += 1

    ttk.Label(main_frame, text="Кодек:").grid(row=row, column=0, sticky="w")
    ttk.Combobox(main_frame, textvariable=codec_var, values=["H264", "H265"], state="readonly", width=7)\
        .grid(row=row, column=1, sticky="w")
    row += 1

    ttk.Label(main_frame, text="Latency (ms):").grid(row=row, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=jitter_var, width=10).grid(row=row, column=1, sticky="w")
    row += 1

    ttk.Label(main_frame, text="Canny T1:").grid(row=row, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=t1_var, width=10).grid(row=row, column=1, sticky="w")
    ttk.Label(main_frame, text="Canny T2:").grid(row=row, column=2, sticky="w")
    ttk.Entry(main_frame, textvariable=t2_var, width=10).grid(row=row, column=3, sticky="w")
    row += 1

    ttk.Label(main_frame, text="Scale (%):").grid(row=row, column=0, sticky="w")
    ttk.Entry(main_frame, textvariable=scale_var, width=10).grid(row=row, column=1, sticky="w")

    ttk.Checkbutton(main_frame, text="Контуры (Canny)", variable=edges_var).grid(row=row, column=2, columnspan=2, sticky="w")
    row += 1

    btn_frame = ttk.Frame(main_frame)
    btn_frame.grid(row=row, column=0, columnspan=4, pady=10, sticky="w")

    def status_from_thread(text: str):
        root.after(0, lambda: status_var.set(text))

    def start_stream():
        if root.viewer_thread is not None and root.viewer_thread.is_alive():
            status_var.set("Уже запущено")
            return

        try:
            scale_pct = int(scale_var.get().strip())
            scale_pct = max(10, min(400, scale_pct))
            cfg = {
                "SOURCE_TYPE": source_type_var.get().strip().upper(),
                "RTSP_URL": rtsp_url_var.get().strip(),
                "UDP_PORT": int(udp_port_var.get().strip()),
                "PAYLOAD": int(payload_var.get().strip()),
                "CODEC": codec_var.get().strip().upper(),
                "BUFFER_SIZE": DEFAULT_CONFIG["BUFFER_SIZE"],
                "JITTER_LATENCY_MS": int(jitter_var.get().strip()),
                "USE_OPENCV": True,
                "CANNY_T1": int(t1_var.get().strip()),
                "CANNY_T2": int(t2_var.get().strip()),
                "SCALE_PERCENT": scale_pct,
                "SHOW_EDGES": bool(edges_var.get()),
            }
        except ValueError as e:
            status_var.set(f"Неверное число в настройках: {e}")
            return

        root.stop_event.clear()
        status_var.set("Стрим запускается…")
        root.viewer_thread = threading.Thread(
            target=run_stream,
            args=(cfg, root.stop_event, status_from_thread),
            daemon=True,
        )
        root.viewer_thread.start()

    def stop_stream():
        status_var.set("Останавливаю…")
        root.stop_event.set()

    def on_close():
        root.stop_event.set()
        root.destroy()

    ttk.Button(btn_frame, text="Старт", width=15, command=start_stream).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Стоп", width=15, command=stop_stream).pack(side=tk.LEFT, padx=5)

    row += 1
    ttk.Label(main_frame, textvariable=status_var, foreground="blue").grid(row=row, column=0, columnspan=4, sticky="w")

    root.protocol("WM_DELETE_WINDOW", on_close)
    return root

if __name__ == "__main__":
    gui = create_gui()
    gui.mainloop()
