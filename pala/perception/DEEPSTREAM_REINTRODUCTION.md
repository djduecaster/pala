# DeepStream Detector Reintroduction Notes

PALA intentionally removed local person detection from the active runtime in the
post-V4 reset. The current perception loop reports camera frames, timing, and
source health only. This document preserves the working shape and known caveats
of the removed DeepStream path so it can be reintroduced later as an optional
perception enricher rather than a behavior dependency.

The last implementation is preserved in Git history at commit `2159212`.

## Previous Runtime Shape

The removed detector accepted RGB NumPy frames from `PerceptionNode` and ran a
background, latest-only DeepStream pipeline:

```text
RGB frame
  -> appsrc
  -> videoconvert
  -> nvvideoconvert (GPU conversion on Jetson)
  -> NVMM/NV12 caps
  -> nvstreammux
  -> nvinfer
  -> fakesink
```

`nvinfer` metadata was read from a source-pad probe and converted into pixel
coordinates `(x1, y1, x2, y2)`, confidence, and class ID. The runtime then chose
one primary detection and normalized it into camera-relative coordinates.

## Jetson Requirements

- NVIDIA DeepStream with the `nvinfer`, `nvstreammux`, and `nvvideoconvert`
  plugins available.
- Python GStreamer bindings (`gi`, `Gst 1.0`).
- DeepStream Python bindings (`pyds`).
- A Python environment that can see Jetson system packages. With `uv`, recreate
  the environment using system site packages when needed.
- A valid `nvinfer` configuration and matching model assets.

The previous PeopleNet configuration lived at:

```text
config/deepstream/peoplenet_int8.txt
```

It expected the model, calibration file, labels, TensorRT engine cache, and
DeepStream custom parser paths referenced by that file to exist on the Jetson.
Person class ID was `0`.

## Important Implementation Details

- Input frames must be contiguous RGB arrays.
- `appsrc` caps used the actual frame width and height at 30 FPS.
- Jetson RGB/BGR conversion could fail on VIC, so `nvvideoconvert.compute-hw`
  was set to GPU when supported.
- The stream mux used batch size 1, live source mode, and a 40 ms batched push
  timeout.
- The detector used latest-only input semantics to avoid building a stale queue.
- Detection callbacks were bounded by `PALA_DS_INFER_TIMEOUT_S`, previously 1.0
  second by default. A timeout returned the latest cached detections.
- Pipeline bus errors, warnings, and EOS must be drained and surfaced.
- Shutdown must stop the worker thread and set the GStreamer pipeline to
  `Gst.State.NULL`.

## Known Failure Modes

- Missing `gi`, `Gst`, or `pyds` imports on Jetson.
- The Python virtual environment cannot see system-installed GStreamer modules.
- Missing or stale TensorRT engine files.
- Incorrect relative model paths in the `nvinfer` configuration.
- Missing NVIDIA custom bbox parser library.
- `appsrc` or caps negotiation failure between RGB system memory and NVMM/NV12.
- Inference callbacks never arrive, producing repeated timeout warnings.
- Treating a successful unit test as proof of live Jetson inference.

## Recommended Future Boundary

Reintroduce detection behind an optional enricher interface after the direct
vision-model behavior loop is stable:

```python
class PerceptionEnricher(Protocol):
    def observe(self, frame: np.ndarray, frame_id: int) -> dict[str, object]: ...
```

The base `PerceptionState` should continue to represent capture truth. Optional
enrichers may publish factual observations such as tracked bounding boxes, but
they should not directly select behavior modes or primitives. Behavior should
receive those facts in a clearly separated `local_observations` payload.

## Reintroduction Acceptance Checks

1. Verify camera capture without detection first.
2. Verify the DeepStream pipeline independently with recorded frames.
3. Confirm detection timestamps and frame IDs match the submitted image.
4. Measure inference latency and dropped-frame behavior on the Jetson.
5. Confirm clean shutdown releases the pipeline and worker thread.
6. Run a controlled live test before enabling local observations in behavior.
