from __future__ import annotations

from typing import List, Optional
import threading
import logging
import os

import numpy as np

from .interface import Detection, DetectorInterface

logger = logging.getLogger(__name__)


class DeepStreamDetector(DetectorInterface):
    def __init__(
        self,
        *,
        config_path: Optional[str],
        person_class_id: int = 0,
        conf_threshold: Optional[float] = None,
    ) -> None:
        self._config_path = config_path
        self._person_class_id = int(person_class_id)
        self._conf_threshold = conf_threshold

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._infer_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._has_unread = False
        self._latest_dets: List[Detection] = []
        self._last_error: Optional[str] = None
        self._pipeline = None
        self._appsrc = None
        self._bus = None
        self._gst = None
        self._pyds = None
        self._frame_pts = 0
        self._infer_timeout_s = float(os.getenv("PALA_DS_INFER_TIMEOUT_S", "1.0"))
        self._seen_infer_result = False
        self._warned_timeout = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        with self._lock:
            self._latest_frame = frame
            self._has_unread = True
            self._cond.notify_all()
            if self._last_error is not None:
                raise RuntimeError(self._last_error)
            return list(self._latest_dets)

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)

    def _run(self) -> None:
        if not self._config_path:
            with self._lock:
                self._last_error = "DeepStream config_path is required"
                self._stop.set()
                self._cond.notify_all()
            return

        while not self._stop.is_set():
            with self._lock:
                while not self._stop.is_set() and not self._has_unread:
                    self._cond.wait(timeout=0.05)
                if self._stop.is_set():
                    break
                frame = self._latest_frame
                self._has_unread = False

            if frame is None:
                continue

            try:
                dets = self._infer(frame)
            except Exception as exc:
                with self._lock:
                    self._last_error = repr(exc)
                    self._stop.set()
                    self._cond.notify_all()
                return

            with self._lock:
                self._latest_dets = dets

    def _infer(self, _frame: np.ndarray) -> List[Detection]:
        frame = np.ascontiguousarray(_frame)
        self._ensure_pipeline(frame)
        self._drain_bus()

        buf = self._gst.Buffer.new_allocate(None, frame.nbytes, None)
        buf.fill(0, frame.tobytes())
        duration = self._gst.util_uint64_scale_int(1, self._gst.SECOND, 30)
        buf.pts = self._frame_pts
        buf.duration = duration
        self._frame_pts += duration

        self._infer_event.clear()
        ret = self._appsrc.emit("push-buffer", buf)
        if ret != self._gst.FlowReturn.OK:
            raise RuntimeError(f"DeepStream appsrc push-buffer failed: {ret}")

        signaled = self._infer_event.wait(timeout=self._infer_timeout_s)
        self._drain_bus()
        if signaled:
            if not self._seen_infer_result:
                logger.info("DeepStream inference callback received")
            self._seen_infer_result = True
            self._warned_timeout = False
        elif not self._warned_timeout:
            logger.warning(
                "DeepStream inference timed out after %.2fs; returning latest cached detections",
                self._infer_timeout_s,
            )
            self._warned_timeout = True
        with self._lock:
            return list(self._latest_dets)

    def _ensure_pipeline(self, frame: np.ndarray) -> None:
        if self._pipeline is not None:
            return
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        import pyds

        Gst.init(None)
        self._gst = Gst
        self._pyds = pyds

        height, width = frame.shape[:2]

        pipeline = Gst.Pipeline.new("deepstream-pipeline")
        appsrc = Gst.ElementFactory.make("appsrc", "appsrc")
        videoconvert = Gst.ElementFactory.make("videoconvert", "videoconvert")
        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert")
        capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        pgie = Gst.ElementFactory.make("nvinfer", "pgie")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")

        if not all([pipeline, appsrc, videoconvert, nvvideoconvert, capsfilter, streammux, pgie, sink]):
            raise RuntimeError("DeepStream pipeline element creation failed")
        # RGB/BGR transforms can fail on VIC; force GPU conversion on Jetson.
        if nvvideoconvert.find_property("compute-hw") is not None:
            nvvideoconvert.set_property("compute-hw", 1)

        appsrc.set_property("is-live", True)
        appsrc.set_property("block", True)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=RGB,width={width},height={height},framerate=30/1"
            ),
        )

        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height},framerate=30/1"
            ),
        )

        streammux.set_property("batch-size", 1)
        streammux.set_property("width", width)
        streammux.set_property("height", height)
        streammux.set_property("live-source", 1)
        streammux.set_property("batched-push-timeout", 40000)

        pgie.set_property("config-file-path", self._config_path)

        pipeline.add(appsrc)
        pipeline.add(videoconvert)
        pipeline.add(nvvideoconvert)
        pipeline.add(capsfilter)
        pipeline.add(streammux)
        pipeline.add(pgie)
        pipeline.add(sink)

        if not appsrc.link(videoconvert):
            raise RuntimeError("Failed to link appsrc -> videoconvert")
        if not videoconvert.link(nvvideoconvert):
            raise RuntimeError("Failed to link videoconvert -> nvvideoconvert")
        if not nvvideoconvert.link(capsfilter):
            raise RuntimeError("Failed to link nvvideoconvert -> capsfilter")

        srcpad = capsfilter.get_static_pad("src")
        sinkpad = streammux.get_request_pad("sink_0")
        if srcpad is None or sinkpad is None:
            raise RuntimeError("Failed to get pads for streammux linking")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link capsfilter -> streammux")

        if not streammux.link(pgie):
            raise RuntimeError("Failed to link streammux -> nvinfer")
        if not pgie.link(sink):
            raise RuntimeError("Failed to link nvinfer -> sink")

        pgie_src_pad = pgie.get_static_pad("src")
        if pgie_src_pad is None:
            raise RuntimeError("Failed to get nvinfer src pad")
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_infer, None)

        state_ret = pipeline.set_state(Gst.State.PLAYING)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set DeepStream pipeline to PLAYING")
        self._pipeline = pipeline
        self._appsrc = appsrc
        self._bus = pipeline.get_bus()

    def _on_infer(self, _pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return self._gst.PadProbeReturn.OK
        batch_meta = self._pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        dets: List[Detection] = []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = self._pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj_meta = self._pyds.NvDsObjectMeta.cast(l_obj.data)
                conf = float(obj_meta.confidence)
                if obj_meta.class_id == self._person_class_id:
                    if self._conf_threshold is None or conf >= self._conf_threshold:
                        rect = obj_meta.rect_params
                        dets.append(
                            Detection(
                                bbox_xyxy_px=(
                                    rect.left,
                                    rect.top,
                                    rect.left + rect.width,
                                    rect.top + rect.height,
                                ),
                                conf=conf,
                                cls=int(obj_meta.class_id),
                            )
                        )
                l_obj = l_obj.next
            l_frame = l_frame.next
        with self._lock:
            self._latest_dets = dets
        self._infer_event.set()
        return self._gst.PadProbeReturn.OK

    def _drain_bus(self) -> None:
        if self._bus is None:
            return
        mask = (
            self._gst.MessageType.ERROR
            | self._gst.MessageType.WARNING
            | self._gst.MessageType.EOS
        )
        while True:
            msg = self._bus.pop_filtered(mask)
            if msg is None:
                break
            if msg.type == self._gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                raise RuntimeError(f"GStreamer/DeepStream error: {err}; debug={dbg}")
            if msg.type == self._gst.MessageType.WARNING:
                warn, dbg = msg.parse_warning()
                logger.warning("GStreamer warning: %s; debug=%s", warn, dbg)
            if msg.type == self._gst.MessageType.EOS:
                raise RuntimeError("GStreamer/DeepStream EOS received")
