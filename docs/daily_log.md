# Daily Log

## 2026-02-07

### DeepStream bring-up lessons
- DeepStream engine builds can take multiple minutes on first run; short `PALA_MAX_RUNTIME_S` values can exit before serialization finishes.
- `PerceptionNode` currently falls back to a center dummy bbox when detector output is empty, so `zone=center` does not prove detector success.
- `nvinfer` model paths in config are resolved relative to the config file location; use absolute paths or correct relative paths from `config/deepstream/`.
- On this Jetson setup, `nvvideoconvert` failed with VIC for RGB/BGR transforms; forcing GPU conversion (`compute-hw=1`) is required for this pipeline.
- Treat `Deserialize engine failed` as expected on first run if engine file does not exist yet; focus on subsequent build/serialize logs.

### Python environment lessons
- `jetson_full` requires `smbus2` for PCA9685 control.
- DeepStream Python integration requires both `gi` and `pyds` available inside the `uv` environment.
- `uv venv --system-site-packages` is needed so `uv run` can import system-provided `gi`.
- `pyds` may not ship in the DeepStream `.deb`; install the matching wheel (DS 7.1 -> `pyds` 1.2.0 for cp310 aarch64).
- DeepStream Python bindings currently work more reliably with `numpy<2`; pinning avoids runtime incompatibilities.

### Workflow notes
- For long Jetson diagnostics, always capture output with `tee` and grep for `engine|serialize|error|nvvideoconvert`.
- Keep one temporary, explicit engine path during bring-up to remove ambiguity (`/tmp/...engine` or absolute project path).
