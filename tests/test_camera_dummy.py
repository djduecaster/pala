from pala.hardware.camera import DummyCamera


def test_dummy_camera_frame_shape_and_timestamps():
    cam = DummyCamera(width=320, height=240)
    frame, pts_ns, mono_ns = cam.get_frame()
    assert frame.shape == (240, 320, 3)
    assert frame.dtype.name == "uint8"
    assert pts_ns is None
    assert isinstance(mono_ns, int)
