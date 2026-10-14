"""Sign art: what signs.draw makes of each style, and the rules that its lettering keeps."""
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from sgi.synth.realistic import signs
from sgi.synth.realistic.spec import SIGN_STYLES, SignArt

CREAM, BROWN, PAPER, NAVY, SLATE = (245, 240, 225), (126, 58, 40), (244, 240, 228), (40, 60, 120), (40, 44, 52)

# Each style on a rectangle of the size a scene gives it, with the text a scene gives it.
ART = {
    "label": SignArt("label", "COFFEE", CREAM, BROWN, 1.5, 0.3),
    "logo": SignArt("logo", "coffee", CREAM, BROWN, 1.6, 0.35, seed=3),
    "pictogram": SignArt("pictogram", "information", (255, 255, 255), (30, 80, 160), 0.3, 0.3),
    "plate": SignArt("plate", "TICKETS", SLATE, (236, 234, 226), 0.25, 0.12),
    "menu": SignArt("menu", "coffee", (240, 234, 220), (38, 44, 42), 0.6, 0.9, seed=1),
    "map": SignArt("map", "", NAVY, PAPER, 1.2, 1.2, seed=2),
    "exit": SignArt("exit", "", width_m=0.3, height_m=0.15),
    "notice": SignArt("notice", "events", SLATE, PAPER, 0.3, 0.42, seed=4),
    "screen": SignArt("screen", "START", (226, 236, 244), (22, 30, 44), 0.5, 0.3),
    "paper": SignArt("paper", "", (60, 60, 70), (246, 226, 130), 0.21, 0.3, seed=1),
}
LETTERED = ("plate", "logo", "menu", "map", "notice")          # the natural styles that carry words
WORDLESS = ("pictogram", "exit", "paper")
SEEDED = ("logo", "menu", "map", "notice")                     # the styles whose content the seed draws
SHAPES = ((1.6, 0.35), (0.5, 0.2), (0.25, 0.12), (0.3, 0.42), (0.6, 0.9), (1.2, 1.2), (0.12, 0.4), (2.0, 0.1))
THEMES = ("coffee", "TICKETS", "print", "help", "flowers", "fruit", "sweets", "events", "notices", "", "Lost property")
SMALL = 0.25                                                   # the most of a picture's height that small type takes
TYPE_M = 0.04                                                  # and the tallest type of a menu, a map or a notice

# Real chains that a made-up name must not be, or hold a word of.
REAL_CHAINS = ("Starbucks", "Costa", "Caffe Nero", "Pret A Manger", "Greggs", "McDonald's", "Burger King", "Subway",
               "Dunkin", "Tim Hortons", "Peet's", "Lavazza", "Illy", "Nescafe", "Tchibo", "Krispy Kreme", "KFC",
               "7-Eleven", "Tesco", "Sainsbury's", "Walmart", "Target", "Aldi", "Lidl", "Carrefour", "Whole Foods",
               "Ikea", "Amazon", "Apple", "Google", "FedEx", "UPS", "DHL", "Kinko's", "Staples", "Ticketmaster",
               "Eventbrite", "Odeon", "Cineworld", "Interflora", "Haribo", "Cadbury", "Hershey's", "Lindt", "Boots",
               "WHSmith", "Lowe's", "Bell", "Voss", "Shell", "Nike", "Zara")
FILLER = {"a", "and", "co", "the", "s"}                        # words of a name that are no one's

# A notice in grey on white, whose picture is the one part of it in colour, and the two shapes of sheet that a
# scene pins up: an upright one, with the picture across its head, and one of a notice board's that is wider
# than high, with the picture in a column at its left.
PLAIN = SignArt("notice", "", (40, 40, 40), (255, 255, 255), 0.3, 0.42)
UPRIGHT, LANDSCAPE = (0.3, 0.42), (0.24, 0.2)
# What stands over each heading of a notice. One thing, which has its own outline whatever the sheet's shape:
THINGS = {"a head and shoulders": ("Film night", "Quiz night", "Choir evening", "Volunteers wanted"),
          "a clock": ("Opening hours", "Cleaning rota"),
          "a warning": ("Lift maintenance", "Fire drill", "Car park closure"),
          "an umbrella": ("Lost property",)}
# or a scene, which fills the picture from side to side;
SCENES = ("Summer concert", "Craft fair", "Book sale")
# or either, by the seed.
EITHER = ("Open day", "Spring market", "Meeting room bookings")


def words_of(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", name.lower())) - FILLER


def is_colour(picture: np.ndarray, colour) -> np.ndarray:
    return (picture == np.asarray(colour, dtype=np.uint8)).all(axis=-1)


def nearer(picture: np.ndarray, colour, than) -> np.ndarray:
    """Where the picture is nearer to one colour than to the other."""
    picture = picture.astype(np.float64)
    return np.linalg.norm(picture - colour, axis=-1) < np.linalg.norm(picture - than, axis=-1)


def off_the_line(picture: np.ndarray, a, b) -> np.ndarray:
    """How far each pixel's colour is from every mixture of two colours."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    along = np.clip(((picture - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
    return np.linalg.norm(picture - (a + along[..., None] * (b - a)), axis=-1)


@dataclass(frozen=True)
class Set:
    """One piece of text that Pillow was asked to set."""
    words: str
    cap: float                                     # how high the capitals of its font stand, of the picture's height
    left: float                                    # where its ink begins and ends, of the picture's width
    right: float
    margin: float                                  # the least from its ink to a side, of the picture's shorter side


@pytest.fixture
def lettering(monkeypatch):
    """Every piece of text that Pillow sets from here on, measured on the picture it is set on."""
    seen: list[Set] = []
    set_text = ImageDraw.ImageDraw.text

    def text(self, xy, text, *args, **kwargs):
        font, stroke = kwargs["font"], kwargs.get("stroke_width", 0)
        # The ink as it is drawn: the box that Pillow reports for a text has the white beside its letters in it.
        mask, (x, y) = font.getmask2(text, mode="L", anchor=kwargs.get("anchor"), stroke_width=stroke)
        left, top, right, bottom = (at + start for at, start in zip(mask.getbbox(), (xy[0] + x, xy[1] + y) * 2))
        width, height = self.im.size
        _, cap_top, _, cap_bottom = font.getbbox("H", stroke_width=stroke)
        seen.append(Set(text, (cap_bottom - cap_top) / height, left / width, right / width,
                        min(left, top, width - right, height - bottom) / min(width, height)))
        return set_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    return seen


def words(seen: list[Set]) -> list[Set]:
    """The text that is lettering: a single letter is a logo's monogram, which is its mark."""
    return [one for one in seen if len(one.words.strip()) > 1]


def seeds_by_heading(lettering: list[Set], seeds=range(80)) -> dict[str, list[int]]:
    """The seeds that give a notice each heading. What a seed draws is the same at any size, so they are found
    on small sheets."""
    found: dict[str, list[int]] = {}
    for seed in seeds:
        del lettering[:]
        signs.draw(replace(PLAIN, seed=seed), px_per_m=128)
        found.setdefault(lettering[0].words, []).append(seed)          # the heading is the first text set
    return found


def block_of(notice: np.ndarray) -> tuple[slice, slice]:
    """Where the picture is on a notice in grey on white: the rows and the columns of what is in colour."""
    coloured = notice.max(axis=-1).astype(int) - notice.min(axis=-1) > 12
    rows, columns = np.flatnonzero(coloured.any(axis=1)), np.flatnonzero(coloured.any(axis=0))
    return slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1)


def picture_on(width_m: float, height_m: float, seed: int, px_per_m: int = 1024) -> np.ndarray:
    """The picture of a notice in grey on white, without the pixels of its edge, which are mixed with the sheet's."""
    notice = signs.draw(replace(PLAIN, width_m=width_m, height_m=height_m, seed=seed), px_per_m=px_per_m)
    rows, columns = block_of(notice)
    return notice[rows.start + 1:rows.stop - 1, columns.start + 1:columns.stop - 1]


def drawn_on(picture: np.ndarray) -> np.ndarray:
    """Where a picture is not its background, which is the one colour that most of it has."""
    colours, counts = np.unique(picture.reshape(-1, 3), axis=0, return_counts=True)
    return (np.abs(picture.astype(int) - colours[counts.argmax()]) > 8).any(axis=-1)


def outline(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """What is drawn, apart from where and how large: its width over its height, and its shape on a square."""
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    cut = mask[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    square = Image.fromarray(cut.astype(np.uint8) * 255).resize((32, 32), Image.Resampling.BOX)
    return cut.shape[1] / cut.shape[0], np.asarray(square) > 127


def alike(a: np.ndarray, b: np.ndarray) -> float:
    """How much two shapes on a square have in common: the share of what either covers that both cover."""
    return float((a & b).sum() / (a | b).sum())


def test_every_style_of_the_scene_description_is_drawn():
    assert set(ART) == set(SIGN_STYLES)
    for art in ART.values():
        assert signs.draw(art).std() > 0.0 or art.style == "paper"


@pytest.mark.parametrize("style", SIGN_STYLES)
def test_the_drawing_has_the_proportions_of_its_rectangle(style):
    art = ART[style]
    for px_per_m in (512, 110):
        picture = signs.draw(art, px_per_m=px_per_m)
        assert picture.dtype == np.uint8
        assert picture.shape == (round(art.height_m * px_per_m), round(art.width_m * px_per_m), 3)
    assert signs.draw(art).shape == signs.draw(art, px_per_m=512).shape


@pytest.mark.parametrize("style", SIGN_STYLES)
def test_the_smallest_drawing_is_8_pixels_a_side(style):
    assert signs.draw(replace(ART[style], width_m=0.01, height_m=0.004)).shape == (8, 8, 3)
    assert signs.draw(replace(ART[style], width_m=0.5, height_m=0.004)).shape == (8, 256, 3)


@pytest.mark.parametrize("style", ("label", "menu", "map"))
def test_a_drawing_beyond_4096_pixels_is_scaled_down_in_proportion(style):
    assert signs.draw(replace(ART[style], width_m=20.0, height_m=5.0)).shape == (1024, 4096, 3)
    assert signs.draw(replace(ART[style], width_m=3.0, height_m=12.0)).shape == (4096, 1024, 3)


@pytest.mark.parametrize("style", SIGN_STYLES)
def test_the_same_art_gives_the_same_pixels(style):
    art = ART[style]
    first = signs.draw(art)
    assert np.array_equal(first, signs.draw(replace(art)))
    assert signs.text_heights(art) == signs.text_heights(replace(art))


@pytest.mark.parametrize("style", SIGN_STYLES)
def test_the_drawing_neither_reads_nor_moves_the_global_random_state(style):
    random.seed(1)
    np.random.seed(1)
    first = signs.draw(ART[style])
    random.seed(2)
    np.random.seed(2)
    state, numpy_state = random.getstate(), np.random.get_state()[1].copy()
    assert np.array_equal(first, signs.draw(ART[style]))
    assert random.getstate() == state
    assert np.array_equal(np.random.get_state()[1], numpy_state)


@pytest.mark.parametrize("style", SEEDED)
def test_the_seed_changes_what_is_drawn(style):
    pictures = [signs.draw(replace(ART[style], seed=seed), px_per_m=256) for seed in range(6)]
    assert len({picture.tobytes() for picture in pictures}) == 6


def test_a_label_shows_ink_on_its_ground():
    art = ART["label"]
    picture = signs.draw(art)
    assert is_colour(picture, art.ink).any() and is_colour(picture, art.ground).any()
    assert 0.05 < nearer(picture, art.ink, art.ground).mean() < 0.60
    # Nothing but the two colours and what lies between them at the edge of a letter.
    assert off_the_line(picture, art.ink, art.ground).max() < 3.0


def test_a_label_is_as_large_as_the_placeholder_letters_it():
    """Capitals 0.62 of the board high, no wider than 0.86 of it: what recipes._place_sign asks for."""
    art = SignArt("label", "HELP", CREAM, (30, 110, 100), 1.6, 0.35)
    picture = signs.draw(art)
    rows, columns = np.nonzero(nearer(picture, art.ink, art.ground))
    height, width = picture.shape[:2]
    assert (rows.max() + 1 - rows.min()) / height == pytest.approx(0.62, abs=0.02)
    assert (columns.max() + 1 - columns.min()) / width < 0.86
    assert signs.text_heights(art) == [pytest.approx((rows.max() + 1 - rows.min()) / height, abs=2 / height)]
    # It stands in the middle of its board.
    assert rows.min() + rows.max() + 1 == pytest.approx(height, abs=3)
    assert columns.min() + columns.max() + 1 == pytest.approx(width, abs=3)


def test_long_text_is_shrunk_to_fit_and_never_clipped():
    long = SignArt("label", "INFORMATION AND LEFT LUGGAGE ENQUIRIES", CREAM, BROWN, 1.5, 0.3)
    narrow = replace(long, text="TICKETS", width_m=0.2, height_m=1.0)
    for art in (long, narrow, replace(long, text="Lifts, platforms & parking (ground floor)")):
        picture = signs.draw(art)
        height, width = picture.shape[:2]
        ink = nearer(picture, art.ink, art.ground)
        assert ink.any()
        # A margin of the ground's colour on all four sides, and the ink no wider than the placeholder's 0.86.
        frame = np.ones((height, width), dtype=bool)
        frame[round(0.1 * height):height - round(0.1 * height), round(0.06 * width):width - round(0.06 * width)] = False
        assert is_colour(picture[frame], art.ground).all()
        columns = np.nonzero(ink.any(axis=0))[0]
        assert (columns.max() + 1 - columns.min()) / width <= 0.86 + 1 / width
    (short,), (shrunk,) = signs.text_heights(replace(long, text="HELP")), signs.text_heights(long)
    assert shrunk < 0.5 * short


@pytest.mark.parametrize("style", SIGN_STYLES)
def test_no_text_of_any_style_touches_the_edge(style, lettering):
    for width_m, height_m in ((ART[style].width_m, ART[style].height_m), *SHAPES):
        for text in (ART[style].text, "An unusually long name of a place, with a tail", "gypsy query"):
            if style == "pictogram":
                text = "information"
            del lettering[:]
            signs.draw(replace(ART[style], text=text, width_m=width_m, height_m=height_m), px_per_m=256)
            for one in lettering:
                assert one.margin >= 0.025, (style, width_m, height_m, one)


@pytest.mark.parametrize("style", (*LETTERED, "screen"))
def test_natural_styles_carry_no_large_lettering(style):
    for width_m, height_m in ((ART[style].width_m, ART[style].height_m), *SHAPES):
        for text in THEMES:
            for seed in range(2):
                art = replace(ART[style], text=text, width_m=width_m, height_m=height_m, seed=seed)
                heights = signs.text_heights(art, px_per_m=256)
                if style == "plate" and not text:              # a plate with no word is its pictogram alone
                    assert heights == []
                    continue
                assert heights, f"{art} carries no lettering at all"
                assert max(heights) <= SMALL, f"{art}: lettering {max(heights):.2f} of its height"


@pytest.mark.parametrize("style", SIGN_STYLES)
def test_the_text_heights_are_those_of_the_lettering_that_was_drawn(style, lettering):
    """text_heights is what the rule on lettering is checked by: it has to list all that Pillow set."""
    for width_m, height_m in ((ART[style].width_m, ART[style].height_m), (1.6, 0.35), (0.3, 0.42)):
        for seed in range(4):
            art = replace(ART[style], width_m=width_m, height_m=height_m, seed=seed)
            heights = signs.text_heights(art)
            del lettering[:]
            signs.draw(art)
            assert sorted(heights) == pytest.approx(sorted(one.cap for one in words(lettering)))
            marks = [one for one in lettering if one not in words(lettering)]
            assert len(marks) <= (1 if style == "logo" else 0), marks


@pytest.mark.parametrize("style", WORDLESS)
def test_a_pictogram_an_exit_sign_and_a_sheet_of_paper_carry_no_text(style, lettering):
    names = signs.PICTOGRAMS if style == "pictogram" else (ART[style].text,)
    for name in names:
        for seed in range(4):
            art = replace(ART[style], text=name, seed=seed)
            signs.draw(art)
            assert signs.text_heights(art) == []
    assert lettering == []


def test_the_pictograms_are_the_five_a_scene_asks_for():
    assert signs.PICTOGRAMS == ("information", "ticket", "printer", "cup", "exit")


@pytest.mark.parametrize("name", ("information", "ticket", "printer", "cup", "exit"))
def test_a_pictogram_is_one_symbol_that_fills_its_rectangle(name):
    art = replace(ART["pictogram"], text=name)
    for width_m, height_m in ((0.3, 0.3), (0.4, 0.2), (0.15, 0.3)):
        picture = signs.draw(replace(art, width_m=width_m, height_m=height_m))
        height, width = picture.shape[:2]
        assert is_colour(picture, art.ink).any() and is_colour(picture, art.ground).any()
        assert off_the_line(picture, art.ink, art.ground).max() < 3.0
        rows, columns = np.nonzero(nearer(picture, art.ink, art.ground))
        across, up = columns.max() + 1 - columns.min(), rows.max() + 1 - rows.min()
        assert max(across, up) >= 0.6 * min(width, height)                   # it fills the shorter side
        assert across <= 0.92 * width and up <= 0.92 * height                # and leaves a margin
        assert columns.min() + columns.max() + 1 == pytest.approx(width, abs=0.12 * min(width, height))
        assert rows.min() + rows.max() + 1 == pytest.approx(height, abs=0.12 * min(width, height))
        assert 0.08 < nearer(picture, art.ink, art.ground).sum() / min(width, height) ** 2 < 0.6   # a figure, no speck


def test_the_pictograms_differ_from_one_another():
    art = ART["pictogram"]
    symbols = [nearer(signs.draw(replace(art, text=name)), art.ink, art.ground) for name in signs.PICTOGRAMS]
    assert len(symbols) == 5
    for i, one in enumerate(symbols):
        for other in symbols[i + 1:]:
            assert (one != other).mean() > 0.05


def test_the_name_of_a_pictogram_is_read_whatever_its_case():
    art = ART["pictogram"]
    assert np.array_equal(signs.draw(replace(art, text=" Printer ")), signs.draw(replace(art, text="printer")))


def test_an_unknown_pictogram_is_an_error_that_names_the_five():
    with pytest.raises(ValueError, match="unknown pictogram 'umbrella'.*information, ticket, printer, cup, exit"):
        signs.draw(replace(ART["pictogram"], text="umbrella"))
    with pytest.raises(ValueError, match="unknown pictogram ''"):
        signs.text_heights(replace(ART["pictogram"], text=""))


def test_an_unknown_style_is_an_error_that_names_the_styles():
    with pytest.raises(ValueError, match="unknown style 'banner'.*label, logo, pictogram"):
        signs.draw(SignArt("banner", "HELP"))
    with pytest.raises(ValueError, match="unknown style 'banner'"):
        signs.text_heights(SignArt("banner", "HELP"))


@pytest.mark.parametrize("width_m, height_m", ((0.0, 0.3), (1.0, -0.3), (float("nan"), 0.3), (1.0, float("inf"))))
def test_a_rectangle_without_a_size_is_an_error(width_m, height_m):
    with pytest.raises(ValueError, match="width_m and height_m"):
        signs.draw(SignArt("label", "HELP", width_m=width_m, height_m=height_m))


def test_a_scale_that_is_not_above_zero_is_an_error():
    with pytest.raises(ValueError, match="px_per_m"):
        signs.draw(ART["label"], px_per_m=0)


def test_a_made_up_name_is_not_a_real_brand():
    assert len(signs.NAMES) >= 12 and len(set(signs.NAMES)) == len(signs.NAMES)
    real = set().union(*(words_of(chain) for chain in REAL_CHAINS))
    squeezed = {re.sub(r"[^a-z0-9]", "", chain.lower()) for chain in REAL_CHAINS}
    for name in signs.NAMES:
        assert words_of(name), name
        assert not words_of(name) & real, f"{name!r} holds a word of a real chain"
        assert not any(chain in re.sub(r"[^a-z0-9]", "", name.lower()) for chain in squeezed if len(chain) > 3), name
    # The check sees a real name when there is one.
    assert words_of("Costa & Wren") & real and words_of("Peet's Yard") & real


def test_a_logo_carries_a_made_up_name_that_the_seed_picks(lettering):
    names = []
    for seed in range(24):
        del lettering[:]
        signs.draw(replace(ART["logo"], seed=seed), px_per_m=256)
        named = [one.words for one in lettering if one.words in signs.NAMES]
        assert len(named) == 1, [one.words for one in lettering]
        names.append(named[0])
    assert len(set(names)) >= 6
    # A name is no word for what the place sells: the mark shows that, and nothing spells it out.
    for theme in THEMES:
        assert not any(word in words_of(name) for name in signs.NAMES for word in words_of(theme))


def test_a_logo_keeps_its_name_whatever_its_size_and_its_colours(lettering):
    art = ART["logo"]
    square, turned = replace(art, width_m=0.4, height_m=0.4), replace(art, ink=BROWN, ground=CREAM)
    for other in (square, turned, replace(art, text="COFFEE")):
        del lettering[:]
        signs.draw(art)
        signs.draw(other)
        first, second = (one.words for one in lettering if one.words in signs.NAMES)
        assert first == second


def test_the_mark_of_a_logo_is_larger_than_its_name():
    """The mark may be large and the name is modest: on a header board the picture is most of the ink."""
    art = ART["logo"]
    picture = signs.draw(art)
    rows = np.nonzero(nearer(picture, art.ink, art.ground).any(axis=1))[0]
    assert (rows.max() + 1 - rows.min()) / picture.shape[0] > 2 * max(signs.text_heights(art))


def test_a_plate_shows_its_word_in_sentence_case_beside_a_pictogram(lettering):
    art = ART["plate"]
    picture = signs.draw(art)
    (word,) = lettering
    assert word.words == "Tickets"
    ink = nearer(picture, art.ink, art.ground)
    height, width = ink.shape
    # The pictogram is at one side: ink higher than the word's capitals, outside the columns of the word.
    beside = np.ones(width, dtype=bool)
    beside[int(word.left * width):int(np.ceil(word.right * width))] = False
    rows = np.nonzero(ink[:, beside].any(axis=1))[0]
    assert (rows.max() + 1 - rows.min()) / height > 2 * word.cap
    del lettering[:]
    signs.draw(replace(art, text="left LUGGAGE"))
    assert [one.words for one in lettering] == ["Left luggage"]


def test_a_menu_lists_items_with_prices_in_small_type(lettering):
    for theme in ("coffee", "tickets", "flowers", "fruit", "sweets", "print", "anything else"):
        for width_m, height_m in ((0.6, 0.9), (1.34, 0.48), (0.3, 0.2), (2.0, 1.2)):
            art = replace(ART["menu"], text=theme, width_m=width_m, height_m=height_m)
            del lettering[:]
            signs.draw(art)
            prices = [one for one in lettering if re.fullmatch(r"\d+\.\d\d", one.words)]
            items = [one for one in lettering if one not in prices]
            assert len(prices) >= 3 and len(items) >= len(prices), (theme, width_m, height_m)
            assert max(signs.text_heights(art)) * height_m <= TYPE_M, (theme, width_m, height_m)
            # The rows are set in one size, and the headings in one that is larger.
            assert len({round(one.cap, 6) for one in lettering}) == 2, (theme, width_m, height_m)


@pytest.mark.parametrize("style", ("menu", "map", "notice"))
def test_what_names_no_target_is_set_in_type_no_taller_than_4_cm_however_large_it_is(style):
    """A menu, a map and a notice may be wide, on a target or beside one: what makes them harmless is their type."""
    own = (ART[style].width_m, ART[style].height_m)
    for width_m, height_m in (own, (1.3, 1.2), (2.0, 2.0), (3.0, 1.0), (1.0, 2.4)):
        for text in ("coffee", "tickets", "events", "notices", ""):
            for seed in range(2):
                art = replace(ART[style], text=text, width_m=width_m, height_m=height_m, seed=seed)
                tallest = max(signs.text_heights(art)) * height_m
                assert tallest <= TYPE_M, f"{art}: type {100 * tallest:.1f} cm tall"


def test_a_menu_lists_what_its_theme_sells_in_an_order_that_the_seed_draws(lettering):
    def listed(art: SignArt) -> list[str]:
        del lettering[:]
        signs.draw(art)
        return [one.words for one in lettering if not re.fullmatch(r"\d+\.\d\d", one.words)]

    art = ART["menu"]
    coffee, tickets = listed(art), listed(replace(art, text="tickets"))
    assert not set(coffee) & set(tickets)                    # each theme has its own list
    assert "Tickets" in tickets and "Adult" in tickets
    assert listed(replace(art, seed=art.seed + 1)) != coffee
    assert listed(replace(art, text="COFFEE")) == coffee     # the theme is read whatever its case
    # A larger board of the same art shows more of the same list, and no other one.
    assert set(coffee) <= set(listed(replace(art, width_m=2.0, height_m=1.2)))


def test_a_map_is_a_plan_with_a_marker_and_a_small_legend(lettering):
    art = ART["map"]
    picture = signs.draw(art)
    assert "You are here" in [one.words for one in lettering]
    assert max(signs.text_heights(art)) <= 0.05              # a legend, and no word across the plan
    # The marker is in a colour of its own, which is neither the ink nor the ground nor a tint between them.
    marker = off_the_line(picture, art.ink, art.ground) > 60.0
    assert 0.0005 < marker.mean() < 0.02
    # The plan covers the sheet: walls in ink in every quarter of it.
    ink = nearer(picture, art.ink, art.ground)
    height, width = ink.shape
    for rows in (slice(0, height // 2), slice(height // 2, height)):
        for columns in (slice(0, width // 2), slice(width // 2, width)):
            assert ink[rows, columns].mean() > 0.01


def test_the_exit_sign_is_the_standard_green_with_a_white_figure():
    # Whatever colours the art names: this one sign has the standard's.
    square = replace(ART["exit"], ink=(200, 30, 30), ground=(250, 250, 250), width_m=0.3, height_m=0.3)
    for art in (ART["exit"], square, replace(square, height_m=0.6)):
        picture = signs.draw(art)
        red, green, blue = (picture[..., channel].astype(int) for channel in range(3))
        is_green = (green > red + 40) & (green > blue + 20)
        is_white = (red > 235) & (green > 235) & (blue > 235)
        assert is_green.mean() > 0.4 and 0.1 < is_white.mean() < 0.6
        assert (is_green | is_white).mean() > 0.9            # and nothing else but the edges between them
    assert not np.array_equal(signs.draw(ART["exit"]), signs.draw(replace(ART["exit"], seed=1)))   # the way it points


def test_a_screen_is_light_on_dark_and_its_button_says_the_word(lettering):
    art = ART["screen"]
    picture = signs.draw(art)
    assert "Start" in [one.words for one in lettering]
    assert nearer(picture, art.ground, art.ink).mean() > 0.6
    assert signs.text_heights(replace(art, text=""))         # a screen with no word given still shows an interface


def test_what_a_map_a_notice_and_a_screen_say_of_their_own_names_no_place_of_the_scenes(lettering):
    """A kiosk's screen with a button Help would name the help desk, and a notice headed Tickets the counter."""
    places = {"help", "information", "coffee", "ticket", "tickets", "print", "exit", "desk", "kiosk", "stall",
              "counter"}
    for style, text in (("map", ""), ("notice", "events"), ("notice", "notices"), ("notice", ""), ("screen", "START"),
                        ("screen", "")):
        for seed in range(60):
            signs.draw(replace(ART[style], text=text, seed=seed), px_per_m=128)
    said = set().union(*(words_of(one.words) for one in lettering))
    assert {"here", "stairs", "start", "language"} <= said   # the legend, the buttons: what they say is all seen
    assert not said & places, said & places


def test_a_notice_is_a_sheet_with_a_picture_a_heading_and_lines_of_type(lettering):
    art = ART["notice"]
    picture = signs.draw(art)
    assert len(words(lettering)) >= 1
    assert max(signs.text_heights(art)) <= 0.08
    assert is_colour(picture, art.ground).mean() > 0.3       # a sheet of the ground's colour
    # The picture block: a part of the sheet in colours that are no mixture of the ink and the ground.
    assert (off_the_line(picture, art.ink, art.ground) > 25.0).mean() > 0.05
    # The lines of type: rows of grey between the ground and the ink, many times down the sheet.
    grey = nearer(picture, np.add(art.ink, art.ground) / 2, art.ground)
    grey &= off_the_line(picture, art.ink, art.ground) < 6.0
    bands = np.diff((grey.mean(axis=1) > 0.2).astype(int))
    assert (bands == 1).sum() >= 4


def test_every_heading_of_a_notice_is_met_and_stands_by_a_picture_in_tones_of_its_own(lettering):
    seeds = seeds_by_heading(lettering)
    assert set(seeds) == {*SCENES, *EITHER, *(heading for headings in THINGS.values() for heading in headings)}
    for heading, found in seeds.items():
        for width_m, height_m in (UPRIGHT, LANDSCAPE):
            picture = picture_on(width_m, height_m, found[0], px_per_m=512)
            _, counts = np.unique(picture.reshape(-1, 3), axis=0, return_counts=True)
            assert (counts > 0.005 * counts.sum()).sum() >= 3, (heading, width_m, height_m)
        # And no picture is too much for the smallest sheet, or for a strip.
        assert signs.draw(replace(PLAIN, width_m=0.01, height_m=0.004, seed=found[0])).shape == (8, 8, 3)
        assert signs.draw(replace(PLAIN, width_m=0.5, height_m=0.004, seed=found[0])).shape == (8, 256, 3)


def test_the_thing_on_a_notice_is_the_same_thing_on_a_landscape_sheet_as_on_an_upright_one(lettering):
    """On a landscape sheet the picture is a column at the left, and a head and shoulders was once drawn to the
    column's height: its shoulders an egg, the whole a bowling pin. A thing has its own proportions."""
    seeds = seeds_by_heading(lettering)
    for thing, headings in THINGS.items():
        for heading in headings:
            for seed in seeds[heading][:2]:
                masks = [drawn_on(picture_on(width_m, height_m, seed)) for width_m, height_m in (UPRIGHT, LANDSCAPE)]
                (wide, upright), (tall, landscape) = (outline(mask) for mask in masks)
                assert tall == pytest.approx(wide, rel=0.05), (thing, heading, seed)        # its width over its height
                assert alike(upright, landscape) > 0.8, (thing, heading, seed)
                for mask in masks:                                   # all of it is in the picture: none of it cut off
                    assert not (mask[0].any() or mask[:, 0].any() or mask[:, -1].any()), (thing, heading, seed)
                    # A head and shoulders stands on the foot of the picture, with the head clear above them.
                    assert mask[-1].any() == (thing == "a head and shoulders"), (thing, heading, seed)


def test_the_picture_on_a_notice_is_drawn_for_its_heading(lettering):
    """A car park's closure under a portrait, a fire drill under coloured blocks: the picture was once drawn
    whatever the heading said. Now the headings of one matter show one thing, and those of another, another."""
    seeds = seeds_by_heading(lettering)
    shapes = {thing: [outline(drawn_on(picture_on(*UPRIGHT, seed)))[1] for heading in headings
                      for seed in seeds[heading][:3]] for thing, headings in THINGS.items()}
    for thing, same in shapes.items():
        assert len(same) >= 3
        assert min(alike(same[0], other) for other in same[1:]) > 0.9, thing
        for another, others in shapes.items():
            if another != thing:
                assert max(alike(one, other) for one in same for other in others) < 0.8, (thing, another)


def test_a_scene_on_a_notice_fills_its_picture_from_side_to_side_whatever_the_shape_of_the_sheet(lettering):
    seeds = seeds_by_heading(lettering)
    for heading in SCENES:
        for seed in seeds[heading][:2]:
            for width_m, height_m in (UPRIGHT, LANDSCAPE):
                picture = picture_on(width_m, height_m, seed)
                tenths = np.array_split(drawn_on(picture), 10, axis=1)
                assert all(tenth.mean() > 0.03 for tenth in tenths), (heading, seed, width_m, height_m)
                _, counts = np.unique(picture.reshape(-1, 3), axis=0, return_counts=True)
                assert (counts > 0.01 * counts.sum()).sum() >= 4, (heading, seed, width_m, height_m)


@pytest.mark.parametrize("width_m, height_m", (LANDSCAPE, (0.297, 0.21), (0.42, 0.3)))
def test_the_picture_of_a_landscape_notice_is_no_tall_strip_and_has_a_caption_under_it(width_m, height_m):
    """The column at the left of such a sheet is about twice as high as wide, which frames no picture well."""
    for seed in range(4):
        art = replace(PLAIN, width_m=width_m, height_m=height_m, seed=seed)
        notice = signs.draw(art)
        rows, columns = block_of(notice)
        wide, high = columns.stop - columns.start, rows.stop - rows.start
        assert columns.stop < 0.45 * notice.shape[1]                 # at the left, the heading and the text beside it
        assert wide <= high <= 1.25 * wide, (seed, wide, high)
        # The caption: lines of grey under the picture, as wide as it is, down to the foot of the sheet.
        under = notice[rows.stop + 1:, columns]
        grey = nearer(under, np.add(art.ink, art.ground) / 2, art.ground)
        lines = np.flatnonzero(np.diff((grey.mean(axis=1) > 0.2).astype(int)) == 1)
        assert len(lines) >= 2, seed
        assert lines[-1] > 0.6 * len(under), seed


def test_the_picture_of_a_wide_notice_is_as_high_as_the_column_it_stands_in():
    notice = signs.draw(replace(PLAIN, width_m=0.6, height_m=0.25, seed=1))
    rows, columns = block_of(notice)
    assert rows.stop - rows.start > 0.8 * notice.shape[0]
    assert columns.stop - columns.start >= rows.stop - rows.start


def test_paper_is_a_sheet_tinted_by_its_ground():
    sheets = set()
    for seed in range(8):
        art = replace(ART["paper"], seed=seed)
        picture = signs.draw(art)
        assert is_colour(picture, art.ground).mean() > 0.7
        assert np.abs(picture.reshape(-1, 3).mean(axis=0) - art.ground).max() < 12.0
        sheets.add(picture.tobytes())
    assert len(sheets) == 3                                  # blank, ruled, ruled with a margin
    blank = np.empty_like(picture)
    blank[:] = art.ground
    assert blank.tobytes() in sheets


def test_colours_and_sizes_that_numpy_made_draw_the_same_picture():
    """A scene description may hold numpy's numbers. A tint mixed of two unsigned bytes must not wrap round."""
    for style in ("label", "plate", "menu", "map", "notice", "screen", "paper"):
        for kind in (np.uint8, np.int64):
            art = replace(ART[style], seed=2)
            ink, ground = (tuple(np.array(colour, dtype=kind)) for colour in (art.ink, art.ground))
            made = replace(art, ink=ink, ground=ground, width_m=np.float64(art.width_m),
                           height_m=np.float64(art.height_m), seed=np.int64(art.seed))
            assert np.array_equal(signs.draw(made), signs.draw(art)), (style, kind)


def test_a_font_file_is_used_when_one_is_given(tmp_path):
    """Pillow's built-in font written to a file is a font file like any other, and no test may fetch one."""
    font = ImageFont.load_default(20)
    data = getattr(font, "font_bytes", None) or (font.path.getvalue() if hasattr(font.path, "getvalue") else None)
    if data is None:
        pytest.skip("this Pillow does not give the bytes of its built-in font")
    builtin = tmp_path / "builtin.ttf"
    builtin.write_bytes(data)
    for style in ("label", "logo", "menu"):
        assert np.array_equal(signs.draw(ART[style], font=builtin), signs.draw(ART[style]))
        assert signs.text_heights(ART[style], font=builtin) == signs.text_heights(ART[style])
    with pytest.raises(OSError):
        signs.draw(ART["label"], font=tmp_path / "missing.ttf")
    with pytest.raises(OSError):
        signs.text_heights(ART["label"], font=tmp_path / "missing.ttf")


def test_another_font_gives_other_letters():
    matplotlib = pytest.importorskip("matplotlib")           # it ships DejaVu Sans; nothing is downloaded
    font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    if not font.is_file():
        pytest.skip("this matplotlib has no DejaVu Sans")
    art = ART["label"]
    assert not np.array_equal(signs.draw(art, font=font), signs.draw(art))
    assert np.array_equal(signs.draw(art, font=font), signs.draw(art, font=font))
    assert signs.draw(replace(ART["pictogram"], text="cup"), font=font).tobytes() == \
        signs.draw(replace(ART["pictogram"], text="cup")).tobytes()        # a pictogram is drawn from shapes alone
