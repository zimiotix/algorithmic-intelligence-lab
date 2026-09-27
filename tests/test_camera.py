import math

import numpy as np

from ailab.render.camera import Camera


def world_to_screen(cam, x, y, w, h):
    (sx, sy), (ox, oy), (c, s) = cam.uniforms(w, h)
    rx, ry = c * x - s * y, s * x + c * y
    ndc = (rx * sx + ox, ry * sy + oy)
    return (ndc[0] + 1) * w / 2, (1 - ndc[1]) * h / 2


def test_screen_world_round_trip_with_rotation_zoom_and_pan():
    cam = Camera((0, 0, 160, 90))
    cam.rotation, cam.zoom, cam.center = 0.7, 2.5, (40.0, 30.0)
    for px, py in ((0, 0), (800, 450), (1279, 719), (321, 88)):
        x, y = cam.to_world(px, py, 1280, 720)
        assert np.allclose(world_to_screen(cam, x, y, 1280, 720), (px, py), atol=1e-6)


def test_zoom_keeps_the_point_under_the_cursor():
    cam = Camera((0, 0, 160, 90))
    cam.rotation = -1.1
    before = cam.to_world(300, 200, 1280, 720)
    cam.zoom_at(1.7, 300, 200, 1280, 720)
    assert np.allclose(cam.to_world(300, 200, 1280, 720), before, atol=1e-9)


def test_chase_camera_points_the_heading_up_the_screen():
    cam = Camera((0, 0, 160, 90))
    heading = 2.3
    cam.follow(50.0, 20.0, heading, lead=0.0, k=1.0)
    x, y = 50.0, 20.0
    a = world_to_screen(cam, x, y, 1280, 720)
    b = world_to_screen(cam, x + math.cos(heading), y + math.sin(heading), 1280, 720)
    d = np.subtract(b, a)
    assert abs(d[0]) < 1e-6 and d[1] < 0          # straight up on screen (y grows down)
