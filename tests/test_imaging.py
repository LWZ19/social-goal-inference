import io
import numpy as np
import pytest
from PIL import Image
from sgi.imaging import draw_people_numbers, draw_point, draw_target_marker, encode_png
from sgi.types import DetectedPerson

GREY = np.full((480, 640, 3), 128, np.uint8)
CAMERA = (0.0, 0.0)                                 # where the camera stands on the ground: a person at (d, 0) is d from it

def test_encode_png_refuses_a_side_limit_below_one_pixel():
    for max_side in (0, -5):                                          # written to mean "no resizing": that is None
        with pytest.raises(ValueError, match=f"max_side.*{max_side}"):
            encode_png(GREY, max_side=max_side)
    smallest = encode_png(GREY, max_side=1)                           # the smallest limit there is
    assert (smallest.width, smallest.height) == (1, 1)

def test_encode_png_is_lossless_and_reports_the_size_as_sent():
    rgb = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    part = encode_png(rgb)
    assert (part.width, part.height) == (640, 480) and part.png == encode_png(rgb).png      # same bytes, same cache key
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(part.png))), rgb)
    for max_side, size in [(320, (320, 240)), (333, (333, 250)), (2000, (640, 480))]:      # never upscaled
        small = encode_png(rgb, max_side=max_side)
        assert (small.width, small.height) == size == Image.open(io.BytesIO(small.png)).size

def test_markers_are_drawn_on_a_copy_near_the_point():
    out = draw_target_marker(GREY, 400.0, 300.0)
    assert out is not GREY and out.shape == GREY.shape and out.dtype == np.uint8 and (GREY == 128).all()
    assert tuple(out[300, 416]) == (255, 0, 255) and tuple(out[300, 400]) == (128, 128, 128)   # a hollow ring
    rows, cols = np.nonzero((out != GREY).any(axis=2))
    assert len(rows) > 600                                            # the ring alone is about 290 pixels: the word is drawn
    assert np.hypot(cols + 0.5 - 400.0, rows + 0.5 - 300.0).max() < 90.0
    assert (draw_target_marker(GREY, 1.0, 1.0) != GREY).any(axis=2).sum() > 600      # in a corner the word stays in the image
    dot = draw_point(GREY, 200.5, 100.5, color=(0, 255, 0), label="goal")
    assert tuple(dot[100, 200]) == (0, 255, 0) and (GREY == 128).all()

# ------------------------------------------------------------------------------ the numbers drawn on the people
def body(number, u, v, away):                       # a person whose body point is the pixel (u, v), `away` metres from the camera
    return DetectedPerson(number=number, xy=(away, 0.0), pixel=(u, v))

def marks(drawn, on=GREY):                          # (left, top, right, bottom) of each mark, from the left: what changed, split at clear columns
    changed = (drawn != on).any(axis=2)
    cols = np.nonzero(changed.any(axis=0))[0]
    found = []
    for run in np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1):
        rows = np.nonzero(changed[:, run[0]:run[-1] + 1].any(axis=1))[0]
        found.append((int(run[0]), int(rows[0]), int(run[-1]), int(rows[-1])))
    return found

def disc(u, v, radius=9):                           # the pixels a disc of that radius on (u, v) covers
    return (u - radius, v - radius, u + radius, v + radius)

def same(a, b, box):                                # whether two pictures agree inside a box
    left, top, right, bottom = box
    return np.array_equal(a[top:bottom + 1, left:right + 1], b[top:bottom + 1, left:right + 1])

def test_two_people_twenty_pixels_apart_each_carry_their_number_on_their_body_point():
    one, two = body(1, 300.0, 200.0, 4.0), body(2, 320.0, 200.0, 5.0)
    out = draw_people_numbers(GREY, [one, two], CAMERA)
    assert out is not GREY and out.shape == GREY.shape and out.dtype == np.uint8 and (GREY == 128).all()
    assert marks(out) == [disc(300, 200), disc(320, 200)]            # 9 pixels in radius, a pixel clear of one another: neither is moved
    alone = [draw_people_numbers(GREY, [person], CAMERA) for person in (one, two)]
    assert same(out, alone[0], disc(300, 200)) and same(out, alone[1], disc(320, 200))       # each number is where its person is,
    swapped = draw_people_numbers(GREY, [body(2, 300.0, 200.0, 4.0), body(1, 320.0, 200.0, 5.0)], CAMERA)
    assert not same(out, swapped, disc(300, 200)) and not same(out, swapped, disc(320, 200))  # and the two numbers do not look alike
    fill, ink = tuple(out[200, 293]), tuple(out[200, 291])           # just inside the rim, and the rim
    assert fill != ink and fill != (128, 128, 128) and (out[191:210, 291:310] == fill).all(axis=2).sum() > 150    # a filled disc
    digit = (out[193:208, 293:308] == ink).all(axis=2)               # the number, in the rim's colour, about the middle of the disc
    rows, cols = np.nonzero(digit)
    assert digit.sum() > 8 and abs(cols.mean() + 293 - 300) < 2.5 and abs(rows.mean() + 193 - 200) < 2.5

@pytest.mark.parametrize("shape, radius", [((480, 640), 9), ((960, 1280), 18), ((640, 480), 9), ((300, 200), 8), ((100, 400), 8)])
def test_a_disc_is_a_share_of_the_short_side_of_the_image_and_never_under_eight_pixels(shape, radius):
    blank = np.full((*shape, 3), 128, np.uint8)
    u, v = shape[1] // 2, shape[0] // 2
    for number in (7, 40):                                           # two digits stay inside the disc as one does
        assert marks(draw_people_numbers(blank, [body(number, float(u), float(v), 3.0)], CAMERA), blank) == [disc(u, v, radius)]

def test_a_disc_that_would_touch_one_drawn_before_it_moves_up_by_its_diameter_with_a_line_to_the_body_point():
    near, far = body(1, 400.0, 300.0, 3.0), body(2, 410.0, 300.0, 6.0)          # 10 pixels apart: the discs would overlap
    out = draw_people_numbers(GREY, [near, far], CAMERA)
    assert marks(out) == [(391, 273, 419, 309)]                      # the two discs, one 18 pixels over the other's row
    assert same(out, draw_people_numbers(GREY, [near], CAMERA), (391, 292, 407, 309))    # the nearer keeps its place, left of the line
    moved_to = (401, 273, 419, 290)                                  # 18 pixels up: 20.6 from the other disc's middle, and clear
    assert same(out, draw_people_numbers(GREY, [body(2, 410.0, 282.0, 6.0)], CAMERA), moved_to)      # the number went with the disc
    line = out[292:301, 410]                                         # from under the moved disc down to the body point, outside both
    assert (line != 128).any(axis=1).all() and (out[292:301, 414:420] == 128).all() and (out[302:306, 410:420] == 128).all()
    first = draw_people_numbers(GREY, [body(1, 400.0, 300.0, 6.0), body(2, 410.0, 300.0, 3.0)], CAMERA)     # the other one is nearer:
    assert same(first, draw_people_numbers(GREY, [body(2, 410.0, 300.0, 3.0)], CAMERA), (411, 291, 419, 309))    # now it stays
    assert same(first, draw_people_numbers(GREY, [body(1, 400.0, 282.0, 6.0)], CAMERA), (391, 273, 409, 290))
    assert np.array_equal(draw_people_numbers(GREY, [far, near], CAMERA), out)   # by nearness, whatever order they are given in
    touching = draw_people_numbers(GREY, [body(1, 400.0, 300.0, 3.0), body(2, 417.9, 300.0, 6.0)], CAMERA)
    assert not same(touching, GREY, disc(418, 282)) and same(touching, GREY, (421, 292, 427, 309))        # under a diameter apart: moved
    clear = draw_people_numbers(GREY, [body(1, 400.0, 300.0, 3.0), body(2, 418.0, 300.0, 6.0)], CAMERA)
    assert same(clear, GREY, disc(418, 280)[:2] + (427, 290)) and not same(clear, GREY, (410, 291, 427, 309))   # a diameter apart: clear

def test_people_on_one_pixel_are_stacked_upwards_from_the_nearest_and_none_leaves_the_image():
    stack = [body(1, 200.0, 240.0, 5.0), body(2, 200.0, 240.0, 3.0), body(3, 200.0, 240.0, 4.0)]
    out = draw_people_numbers(GREY, stack, CAMERA)
    assert marks(out) == [(191, 240 - 36 - 9, 209, 249)]            # three discs, each a diameter over the one before
    for number, v, away in ((2, 240.0, 3.0), (3, 222.0, 4.0), (1, 204.0, 5.0)):
        inside = (192, int(v) - 8, 208, int(v) + 8)                  # clear of the rims the discs share
        assert same(out, draw_people_numbers(GREY, [body(number, 200.0, v, away)], CAMERA), inside), number
    top = draw_people_numbers(GREY, [body(1, 100.0, 30.0, 3.0), body(2, 100.0, 30.0, 4.0), body(3, 100.0, 30.0, 5.0)], CAMERA)
    assert marks(top) == [(91, 3, 109, 39)]                          # one move up, to row 12; a second would leave the image
    assert draw_people_numbers(GREY, [], CAMERA) is not GREY and np.array_equal(draw_people_numbers(GREY, [], CAMERA), GREY)
