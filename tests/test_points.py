import pytest
from pydantic import ValidationError
from sgi.types import Point
from sgi.vlm.points import POINT_FORMATS

N, P = POINT_FORMATS["xy_1000"], POINT_FORMATS["xy_pixel"]

def test_both_formats_convert_both_ways_for_640x480():
    assert (N.name, P.name) == ("xy_1000", "xy_pixel")
    for fmt, wire, point in [(N, {"x": 625, "y": 250}, Point(x=625, y=250)), (P, {"x": 400, "y": 120}, Point(x=625, y=250)),
                             (P, {"x": 124, "y": 457}, Point(x=194, y=952)), (P, {"x": 640, "y": 480}, Point(x=1000, y=1000))]:
        assert fmt.to_internal(wire, 640, 480) == point and fmt.from_internal(point, 640, 480) == wire
    assert P.from_internal(Point(x=334, y=668), 640, 480) == {"x": 214, "y": 321}        # rounded, not truncated

@pytest.mark.parametrize("name, bad", [
    ("xy_1000", {"x": 1001, "y": 500}), ("xy_1000", {"x": 500, "y": -1}), ("xy_1000", {"x": 1, "y": 2, "z": 3}),
    ("xy_pixel", {"x": 641, "y": 240}), ("xy_pixel", {"x": 320, "y": 481}), ("xy_pixel", {"x": 320.5, "y": 240})])
def test_wire_point_outside_the_image_fails_validation(name, bad):
    model = POINT_FORMATS[name].wire_model(640, 480)
    assert model is POINT_FORMATS[name].wire_model(640, 480)          # one class per size
    model.model_validate({"x": 0, "y": 480})                          # the edges are inside
    with pytest.raises(ValidationError):
        model.model_validate(bad)

def test_instruction_states_the_range_of_its_format():
    a, b = N.instruction(640, 480), P.instruction(640, 480)
    assert '{"x"' in a and "1000" in a and "640" not in a
    assert '{"x"' in b and "640" in b and "480" in b and "1000" not in b
