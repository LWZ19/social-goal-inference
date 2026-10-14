import math

import numpy as np
import pytest

import sgi.polygons as pg

SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
C = (1.0, 2.0)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


def test_point_in_polygon_counts_the_boundary():
    for p in [(0.5, 1.75), (2.0, 1.0), (2.0, 2.0), (2.0 + 5e-10, 1.0)]:   # inside, edge, vertex, within _EPS
        assert pg.point_in_polygon(p, SQUARE) is True
    for p in [(3.0, 1.0), (-1.0, 1.0), (2.0 + 1e-6, 1.0)]:               # from (-1, 1) the ray crosses the closing edge
        assert pg.point_in_polygon(p, SQUARE) is False
    assert pg.point_in_polygon(np.array([0.5, 1.75]), [[0, 0], [2, 0], [2, 2], [0, 2]]) is True


def test_distances_and_bounds():
    assert pg.distance_to_polygon((0.5, 1.75), SQUARE) == 0.0
    assert pg.distance_to_polygon((2.0, 1.0), SQUARE) == 0.0
    assert pg.distance_to_polygon((3.0, 3.0), SQUARE) == pytest.approx(1.4142135623730951)
    assert pg.distance_to_boundary((0.25, 1.0), SQUARE) == pytest.approx(0.25)   # nearest: the closing edge (0, 2)-(0, 0)
    assert pg.distance_to_boundary((3.0, 1.0), SQUARE) == pytest.approx(1.0)
    assert pg.polygon_bounds([(0.5, 6.0), (1.5, 6.0), (1.5, 7.0), (0.5, 7.0)]) == (0.5, 6.0, 1.5, 7.0)


def test_polygon_that_repeats_its_first_vertex():
    closed = SQUARE + [SQUARE[0]]                                      # zero-length closing edge
    assert pg.distance_to_boundary((0.5, 1.75), closed) == pytest.approx(0.25)
    assert pg.distance_to_polygon((3.0, 3.0), closed) == pytest.approx(1.4142135623730951)


def test_grid_points_in():
    pts = pg.grid_points_in([(1.0, 2.0), (1.2, 2.0), (1.2, 2.2), (1.0, 2.2)], 0.05)
    assert pts.shape == (25, 2) and pts.dtype == np.float64    # 24 * 0.05 == 1.2000000000000002 still counts
    close(pts[[0, 1, 5, 24]], [(1.0, 2.0), (1.0, 2.05), (1.05, 2.0), (1.2, 2.2)])
    off = pg.grid_points_in([(0.01, 0.01), (0.21, 0.01), (0.21, 0.21), (0.01, 0.21)], 0.05)
    assert off.shape == (16, 2) and off.min() == pytest.approx(0.05)   # lattice k * step from the world origin
    assert pg.grid_points_in([(0.01, 0.01), (0.04, 0.01), (0.04, 0.04), (0.01, 0.04)], 0.05).shape == (0, 2)


def test_convex_hull():
    inside, on_edge = (1.0, 0.5), (1.0, 0.0)
    hull = pg.convex_hull([(2.0, 2.0), inside, (0.0, 2.0), on_edge, (2.0, 0.0), (0.0, 0.0), (2.0, 2.0)])
    assert hull == SQUARE                                      # counter-clockwise from the lowest of the leftmost; no point twice, none inside or on an edge
    assert isinstance(hull, list) and all(type(v) is tuple and type(v[0]) is float for v in hull)
    assert pg.convex_hull(np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])) == SQUARE
    dart = [(0.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0)]    # a dart: its notch at (1, 1) is no vertex of the hull
    assert pg.convex_hull(dart) == [(0.0, 0.0), (4.0, 1.0), (1.0, 4.0)]
    circle = pg.disc((5.0, 1.0), 0.5, n=8)                     # a small circle 2.5 m to one side of the square
    both = pg.convex_hull(circle + SQUARE)
    assert len(both) == 9 and both[:2] == [(0.0, 0.0), (2.0, 0.0)] and both[-2:] == [(2.0, 2.0), (0.0, 2.0)]   # the square's four corners, and five of the circle's eight points
    assert (4.5, 1.0) not in both and all(p in both for p in circle[:3] + circle[6:])      # not the three that face the square
    turns = [(b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) for a, b, c in zip(both, both[1:] + both[:1], both[2:] + both[:2])]
    assert min(turns) > 0                                      # every turn to the left
    assert all(pg.point_in_polygon(p, both) for p in circle + SQUARE)
    assert pg.convex_hull([(3.0, 3.0), (1.0, 1.0), (2.0, 2.0), (1.0, 1.0)]) == [(1.0, 1.0), (3.0, 3.0)]      # on one line: its two ends
    assert pg.convex_hull([(1.0, 1.0)] * 3) == [(1.0, 1.0)] and pg.convex_hull([]) == []                      # one point, and none


def test_rectangles():
    close(pg.oriented_rectangle((1.0, 2.0), 0.0, 0.6, 1.6, 1.2), [(1.6, 1.4), (2.6, 1.4), (2.6, 2.6), (1.6, 2.6)])
    r90 = pg.oriented_rectangle((1.0, 2.0), math.pi / 2, 0.9, 1.9, 0.8)
    close(r90, [(1.4, 2.9), (1.4, 3.9), (0.6, 3.9), (0.6, 2.9)])
    assert isinstance(r90, list) and all(type(v) is tuple and type(v[0]) is float for v in r90)
    close(pg.segment_rectangle((1.0, 5.5), (1.0, 7.0), 1.2), [(1.6, 5.5), (1.6, 7.0), (0.4, 7.0), (0.4, 5.5)])
    close(pg.segment_rectangle((1.0, 2.0), (4.0, 6.0), 1.0), [(1.4, 1.7), (4.4, 5.7), (3.6, 6.3), (0.6, 2.3)])
    with pytest.raises(ValueError):
        pg.segment_rectangle((1.0, 2.0), (1.0, 2.0), 1.2)


def test_disc_and_keyhole_ring():
    d = pg.disc(C, 0.5, n=8)
    close(d[:3], [(1.5, 2.0), (1.3535533906, 2.3535533906), (1.0, 2.5)])
    assert len(d) == 8 and len(pg.disc(C, 0.5)) == 48
    r = pg.ring(C, 0.6, 1.6)
    assert len(r) == 146 and r[0] == r[72] == (2.6, 2.0) and r[73] == r[145] == (1.6, 2.0)
    assert r[1][1] > 2.0 > r[74][1]                            # outer counter-clockwise, then back round the inner one
    assert pg.point_in_polygon((1.4, 2.3), r) is False         # hole
    assert pg.point_in_polygon(C, r) is False                  # hole centre: the ray runs along the seam
    assert pg.point_in_polygon((1.0, 3.1), r) is True          # band
    assert pg.point_in_polygon((2.1, 2.0), r) is True          # on the seam, which lies in the band
    assert pg.distance_to_polygon(C, r) == pytest.approx(0.5994289329, abs=1e-9)
    assert pg.distance_to_boundary((1.0, 3.1), r) == pytest.approx(0.4995241108, abs=1e-9)
    assert pg.distance_to_boundary((2.1, 2.0), r) == pytest.approx(0.0, abs=1e-12)   # the seam is an edge
    row = [x for x, y in pg.grid_points_in(r, 0.1) if abs(y - 2.0) < 1e-9]
    assert len(row) == 22                                      # x = -0.6 .. 0.4 and 1.6 .. 2.6, not the hole


def test_ring_sectors():
    sectors = pg.ring_sectors(C, 0.8, 1.7, lambda x, y: x > 2.0)       # an arc that wraps through angle 0
    assert [len(s) for s in sectors] == [30]                   # cells 65..71 and 0..6: one run of 14, not two
    s = sectors[0]
    close([s[0], s[15]], [(2.3925584753, 1.0249200582), (1.6553216354, 2.4588611491)])   # outer 325, inner 35
    assert pg.point_in_polygon((2.25, 2.0), s) and not pg.point_in_polygon((1.5, 2.0), s)
    assert not pg.point_in_polygon((1.8, 2.8), s)              # at 45 degrees: outside the arc
    assert pg.point_in_polygon((1.9, 2.5), s)                  # keep is false here: a kept cell is the whole band
    two = pg.ring_sectors(C, 0.8, 1.7, lambda x, y: abs(y - C[1]) < 0.5)
    assert [len(p) for p in two] == [22, 22]                   # first cell 31, then first cell 67 (wraps)
    assert pg.point_in_polygon((-0.25, 2.0), two[0]) and pg.point_in_polygon((2.25, 2.0), two[1])
    assert pg.ring_sectors(C, 0.8, 1.7, lambda x, y: True) == [pg.ring(C, 0.8, 1.7, n=72)]
    assert pg.ring_sectors(C, 0.8, 1.7, lambda x, y: False) == []
    assert [len(p) for p in pg.ring_sectors(C, 0.8, 1.7, lambda x, y: x > C[0], step_deg=10.0)] == [38]
