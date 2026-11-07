"""Sign art: the pictures on the boards, plates, screens and sheets of the realistic scenes, drawn with Pillow.

draw(art) gives the picture of one SignArt as pixels, with the proportions of the rectangle it fills. The
styles are of two kinds:

- `label` is the placeholder's words, large, on a plain ground: capitals up to 0.62 of the board's height and
  ink no wider than 0.86 of its width, as the placeholder letters its sign boards. It exists for the control
  sets alone, where what the picture says has to equal the placeholder's.
- Every other style is what a real place has, and none of them spells anything out in large letters:

      logo       an invented brand: a mark and a made-up name of NAMES, with its trade in small type below.
                 `text` is the theme (as "coffee"), which chooses the mark; `seed` picks the name
      pictogram  one symbol that fills the rectangle; `text` names it, one of PICTOGRAMS
      plate      a small plate: a pictogram at one side and `text` beside it, in sentence case
      menu       a menu or a price list: small headings over rows of items and prices, by `text` and `seed`
      map        a floor plan with a "you are here" marker and a small legend, by `seed`
      exit       the emergency exit sign: the running figure in a doorway and an arrow. It is the one style
                 whose colours are the standard's and not the art's; an odd `seed` points it the other way
      notice     a printed sheet: a heading in modest type, a picture that suits the heading, and lines of
                 text as grey bars. `text` is "events" or "notices", the kind of sheet; `seed` picks the rest
      screen     what a kiosk or a till shows: a title bar, a prompt, buttons and a status line; `text` is
                 the word on the first button
      paper      a blank or lightly ruled sheet in the ground's colour

`ink` and `ground` are a picture's two main colours, and its tints are mixtures of the two. What has a
colour of its own says something by it or is a picture: the exit sign, the red "you are here" marker of a
map, the green lamp of a screen, and the picture on a notice.

Lettering is no higher than SMALL_TYPE of a picture in any style but `label`, whatever the rectangle's
shape, and on a menu, a map and a notice it is no higher than 4 cm however large they are. No text enters a
margin at the picture's sides: it is set smaller until it fits. text_heights(art) lists what was set, so that
the rules can be checked on the lettering and not on the pictures: a pictogram and a logo's mark may be large,
and a monogram is a mark.

The same art and font give the same pixels: what a seed draws comes from a random.Random of the art's own,
and text is laid out by Pillow's basic engine, which every build of it has.
"""
from __future__ import annotations

import colorsys
import math
import random
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from sgi.synth.realistic.spec import SIGN_STYLES, SignArt
from sgi.synth.render import Color

PICTOGRAMS = ("information", "ticket", "printer", "cup", "exit")
# The brands of the realistic scenes. Each is made up: a family name that no chain carries, and a plain noun.
# None says what its place sells, so a logo's name spells no target out; its mark shows the trade.
NAMES = ("Tolliver & Wren", "Marrick Yard", "Penhallow & Co", "Wexcombe Lane", "Dunmore & Pell", "Larkfield Corner",
         "Fennimore House", "Hale & Osgood", "Stroud & Averill", "Elderwick & Sons", "Mossbank Row",
         "Thistlewood & Co", "Quenby's", "Ostler & Pike", "Sable & Finch", "Nettlecombe & Rye")
SMALL_TYPE = 0.25                                  # the most of a picture's height that lettering takes, but a label's
LABEL_HEIGHT, LABEL_WIDTH = 0.62, 0.86             # the most of a board that a label's ink takes: recipes._place_sign

_MIN_PX, _MAX_PX = 8, 4096                         # the sides of a picture
_OVER = 4                                          # a picture is drawn this many times its size and brought down,
_WORK_PX = 4096                                    # for smooth outlines, while its longer side stays within this
_EDGE = 0.03                                       # of a picture's shorter side: the margin that no text enters
_BOLD = 0.022                                      # of a font's size: the stroke round a heading's letters
_HEAD_M, _ITEM_M = 0.036, 0.026                    # the capitals of a heading and of a row at their largest, in metres,
#                                                    on a menu, a map and a notice: under the 4 cm of their rule
_PICTURE_HIGH = 1.2                                # of its width: a notice's picture where it shares a column
_EXIT_GREEN, _EXIT_WHITE = (0, 135, 82), (255, 255, 255)   # the safety colours of an exit sign
_LAMP_GREEN = (71, 209, 110)                       # the lamp of a screen that is ready


@dataclass(frozen=True)
class _Theme:
    """What a place of one trade shows."""
    words: tuple[str, ...]                         # a SignArt's text names the theme by any of these
    marks: tuple[str, ...]                         # the symbols a logo of it may carry; a plate carries the first
    trades: tuple[str, ...] = ()                   # the small line under a logo's name
    sections: tuple[tuple[str, tuple[str, ...]], ...] = ()   # a menu's headings, each with what it lists
    prices: tuple[float, float, float] = (1.0, 20.0, 0.5)    # the least, the most and the step between two


_THEMES = (
    _Theme(("coffee", "cafe", "espresso", "tea"), ("cup",), ("espresso bar", "coffee & tea", "roastery & bakery"),
           (("Coffee", ("Espresso", "Americano", "Flat white", "Cappuccino", "Latte", "Mocha", "Macchiato", "Cortado")),
            ("Tea & more", ("Breakfast tea", "Earl Grey", "Green tea", "Chai latte", "Hot chocolate", "Iced tea")),
            ("Bakery", ("Croissant", "Cinnamon bun", "Banana bread", "Cheese toastie", "Oat cookie", "Flapjack"))),
           (2.2, 4.8, 0.1)),
    _Theme(("tickets", "ticket", "booking", "bookings"), ("ticket",), ("box office", "tickets & bookings"),
           (("Tickets", ("Adult", "Child", "Concession", "Student", "Senior", "Family of four")),
            ("Passes", ("Day pass", "Weekend pass", "Season pass", "Group of ten", "Guided tour")),
            ("Extras", ("Programme", "Cloakroom", "Audio guide", "Reserved seat"))),
           (3.0, 42.0, 0.5)),
    _Theme(("print", "printing", "printer", "copy", "copies"), ("printer",), ("print & copy", "copy point"),
           (("Print & copy", ("A4 black & white", "A4 colour", "A3 black & white", "A3 colour", "Double sided")),
            ("Scan & photo", ("Scan to email", "Photo 10 x 15", "Photo 13 x 18", "Passport photos")),
            ("Finishing", ("Laminating A4", "Spiral binding", "Stapling"))),
           (0.1, 6.0, 0.05)),
    _Theme(("help", "information", "info", "enquiries", "reception"), ("information",),
           ("visitor information", "information point")),
    _Theme(("flowers", "flower", "florist", "plants"), ("flower", "leaf"), ("florist", "flowers & plants"),
           (("Cut flowers", ("Tulips", "Roses", "Lilies", "Sunflowers", "Peonies", "Carnations", "Freesias")),
            ("Bunches", ("Small bunch", "Large bunch", "Seasonal mix", "Hand-tied bouquet")),
            ("Plants", ("Potted herbs", "Orchid", "Fern", "Succulents"))),
           (3.0, 28.0, 0.5)),
    _Theme(("fruit", "fruits", "greengrocer", "vegetables"), ("fruit", "leaf"), ("greengrocer", "fresh fruit daily"),
           (("Fruit", ("Apples", "Pears", "Bananas", "Oranges", "Grapes", "Plums", "Strawberries", "Lemons")),
            ("In season", ("Cherries", "Peaches", "Apricots", "Blueberries", "Melon")),
            ("Juice", ("Orange juice", "Apple juice", "Carrot & ginger"))),
           (0.8, 5.6, 0.1)),
    _Theme(("sweets", "sweet", "candy", "confectionery"), ("sweet",), ("confectionery", "sweets & chocolate"),
           (("Sweets", ("Fudge", "Toffee", "Liquorice", "Jelly beans", "Peppermints", "Caramels", "Nougat")),
            ("Chocolate", ("Milk bar", "Dark bar", "Truffles", "Honeycomb", "Pralines")),
            ("By weight", ("100 g bag", "250 g bag", "Gift box"))),
           (1.0, 9.0, 0.25)),
)
# What a theme without a list of its own shows, and a text that names no theme.
_PLAIN = _Theme((), ("arch", "bars", "leaf", "peaks", "monogram"), (),
                (("Prices", ("Small", "Medium", "Large", "Extra large", "Single", "Pair", "Set of three", "Gift box")),
                 ("Services", ("Standard", "Express", "Collection", "Delivery", "Wrapping")),
                 ("Extras", ("Bag", "Card", "Ribbon", "Receipt copy"))))
_LOCKUPS = ("bare", "disc", "tile")                # how a logo's mark stands: by itself, or cut out of a shape

_HEADINGS = {   # what a notice is headed, by the kind of sheet its text asks for, each with the pictures that suit it
    "events": (("Summer concert", ("hills",)), ("Film night", ("figure",)), ("Craft fair", ("bunting",)),
               ("Book sale", ("bunting",)), ("Open day", ("hills", "bunting", "figure")), ("Quiz night", ("figure",)),
               ("Choir evening", ("figure",)), ("Spring market", ("hills", "bunting"))),
    "notices": (("Opening hours", ("clock",)), ("Lost property", ("umbrella",)), ("Lift maintenance", ("warning",)),
                ("Fire drill", ("warning",)), ("Volunteers wanted", ("figure",)), ("Car park closure", ("warning",)),
                ("Cleaning rota", ("clock",)), ("Meeting room bookings", ("clock", "figure"))),
}
_WHEN = ("Saturday 14 June, 7 pm", "Every Thursday", "From Monday 3 March", "Friday 21 November, 6.30 pm",
         "Weekdays 8 am to 6 pm", "Sunday 9 February, 2 pm")
_FLOORS = ("Ground floor", "Level 1", "Level 2", "Concourse level", "Floor plan")
_SCREEN_TITLES = ("Self-service", "Welcome", "Service point")
_SCREEN_PROMPTS = ("Touch to begin", "Please choose", "Select an option")
_SCREEN_BUTTONS = ("More", "Language", "Back", "Cancel", "Options")    # none names a place of the scenes
_SCREEN_STATUS = ("Ready", "Card and contactless", "Receipt available")


def draw(art: SignArt, font: Path | None = None, px_per_m: int = 512) -> np.ndarray:
    """The picture of this art: H x W x 3 uint8, its first row the top, W = width_m * px_per_m and
    H = height_m * px_per_m to the nearest pixel, each at least 8 and at most 4096; a picture that would be
    larger is scaled down, both sides alike.

    `font` is a .ttf file; without one the text is set in Pillow's built-in scalable font. An unknown style, a
    pictogram that is none of PICTOGRAMS, a rectangle without a size and a scale not above 0 are ValueErrors."""
    return _drawn(art, font, px_per_m)[0]


def text_heights(art: SignArt, font: Path | None = None, px_per_m: int = 512) -> list[float]:
    """How high the capitals stand in each piece of text that draw sets for this art, as a share of the
    picture's height: one number for every line of words, and none for a symbol or a logo's monogram.

    Times `art.height_m` it is the height in metres, unless a side of the picture was raised to 8 pixels and
    the picture has the rectangle's proportions no more."""
    return _drawn(art, font, px_per_m)[1]


def _drawn(art: SignArt, font: Path | None, px_per_m: float) -> tuple[np.ndarray, list[float]]:
    if art.style not in _STYLES:
        raise ValueError(f"unknown style {art.style!r} of sign art: expected one of {', '.join(SIGN_STYLES)}")
    # The colours as Python's own numbers: a scene description may hold ones that numpy made, and a tint mixed
    # of two of its unsigned bytes would wrap round.
    ink, ground = (tuple(int(channel) for channel in colour) for colour in (art.ink, art.ground))
    art = replace(art, ink=ink, ground=ground)
    sheet = _Sheet(art, None if font is None else str(font), px_per_m)
    # Seeded from a string, which Python hashes the same way in every run: what is drawn depends on the style,
    # the text and the seed alone, and not on the size or the colours.
    _STYLES[art.style](sheet, art, random.Random(f"{art.style}|{_key(art.text)}|{art.seed}"))
    return sheet.pixels(), list(sheet.heights)


def _size(art: SignArt, px_per_m: float) -> tuple[int, int]:
    """The picture's width and height in pixels."""
    if not (math.isfinite(art.width_m) and math.isfinite(art.height_m) and art.width_m > 0 and art.height_m > 0):
        raise ValueError(f"width_m and height_m of sign art must be above 0: got {art.width_m} by {art.height_m}")
    if not (math.isfinite(px_per_m) and px_per_m > 0):
        raise ValueError(f"px_per_m must be above 0: got {px_per_m}")
    width, height = art.width_m * px_per_m, art.height_m * px_per_m
    shrink = min(1.0, _MAX_PX / max(width, height))
    return tuple(min(_MAX_PX, max(_MIN_PX, round(side * shrink))) for side in (width, height))


def _key(text: str) -> str:
    """A text as it is looked up: lower case, one space between words."""
    return " ".join(text.lower().split())


def _sentence(text: str) -> str:
    """A text in sentence case, as a plate or a button shows it."""
    return " ".join(text.split()).capitalize()


def _theme(text: str) -> _Theme | None:
    """The theme that a word of the text names, if one does."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return next((theme for theme in _THEMES if words & set(theme.words)), None)


def _mix(a: Color, b: Color, share: float) -> Color:
    """The colour this share of the way from `a` to `b`."""
    return tuple(round(x + (y - x) * share) for x, y in zip(a, b))


def _shade(hue: float, lightness: float, saturation: float) -> Color:
    return tuple(round(255 * channel) for channel in colorsys.hls_to_rgb(hue % 1.0, lightness, saturation))


def _marker(ink: Color) -> Color:
    """The colour of what must stand out from a drawing in this ink: red, or blue beside an ink that is red."""
    hue, lightness, saturation = colorsys.rgb_to_hls(*(channel / 255 for channel in ink))
    reddish = saturation > 0.3 and 0.15 < lightness < 0.85 and min(hue, 1.0 - hue) < 0.09
    return _shade(0.6 if reddish else 0.0, 0.46, 0.78)


# ---- type ----

@lru_cache(maxsize=None)
def _typeface(path: str | None) -> ImageFont.FreeTypeFont:
    """The font of a file, or Pillow's own without one, laid out by the basic engine: the other engine is
    there in some builds of Pillow only, and spaces the letters differently."""
    if path is None:
        return ImageFont.load_default(64).font_variant(layout_engine=ImageFont.Layout.BASIC)
    return ImageFont.truetype(path, 64, layout_engine=ImageFont.Layout.BASIC)


@lru_cache(maxsize=512)                            # a size of a font is some tens of kilobytes
def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    return _typeface(path).font_variant(size=size)


@lru_cache(maxsize=None)
def _capital_share(path: str | None) -> float:
    """How high a capital stands, of the font's size."""
    _, top, _, bottom = _font(path, 1000).getbbox("H")
    return (bottom - top) / 1000


def _ink(font: ImageFont.FreeTypeFont, words: str, stroke: int) -> tuple[int, int, int, int] | None:
    """Where the ink of some words lies, rightwards and downwards of the start of their baseline: left, top,
    right, bottom. None if they leave no ink.

    It is measured on the letters as they are drawn. The box that a font reports runs from where the pen starts
    to where it ends, with the white beside the first letter and the last: text centred by it stands off centre."""
    mask, (x, y) = font.getmask2(words, mode="L", anchor="ls", stroke_width=stroke)
    box = mask.getbbox()
    return None if box is None else (x + box[0], y + box[1], x + box[2], y + box[3])


@lru_cache(maxsize=512)
def _capital(path: str | None, size: int, stroke: int) -> tuple[int, int]:
    """How far a capital of the font at this size stands above the baseline, and how far it reaches below it."""
    _, top, _, bottom = _ink(_font(path, size), "H", stroke) or (0, 0, 0, 0)
    return -top, bottom


@dataclass(frozen=True)
class _Line:
    """One line of words at the size it is to be set in. Its ink is measured from the start of its baseline."""
    words: str
    font: ImageFont.FreeTypeFont
    stroke: int                                    # the pixels added round every letter, for a bolder one
    left: float                                    # the ink: rightwards and downwards of the baseline's start
    top: float
    right: float
    bottom: float
    rise: float                                    # how far a capital stands above the baseline
    sink: float                                    # and reaches below it, which is the stroke

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def cap(self) -> float:
        return self.rise + self.sink

    def baseline(self, top: float, bottom: float) -> float:
        """The baseline that puts its capitals in the middle between two heights."""
        return (top + bottom + self.rise - self.sink) / 2

    def baseline_of_ink(self, top: float, bottom: float) -> float:
        """The baseline that puts its ink, tails and all, in the middle between two heights."""
        return (top + bottom - self.top - self.bottom) / 2


class _Sheet:
    """The picture as it is drawn: `over` times its size, in the ground's colour, with the text set so far."""

    def __init__(self, art: SignArt, font: str | None, px_per_m: float):
        width, height = _size(art, px_per_m)
        self.over = max(1, min(_OVER, _WORK_PX // max(width, height)))
        self.w, self.h = width * self.over, height * self.over
        self.unit = min(self.w, self.h)
        self.m = math.sqrt(self.w * self.h / (art.width_m * art.height_m))    # pixels of this drawing to a metre
        self.edge = max(self.over, math.ceil(_EDGE * self.unit))              # no text within this of a side
        self.most = (1.0 if art.style == "label" else SMALL_TYPE) * self.h    # no capital higher than this
        self.font = font
        self.image = Image.new("RGB", (self.w, self.h), tuple(art.ground))
        self.pen = ImageDraw.Draw(self.image)
        self.heights: list[float] = []

    def pixels(self) -> np.ndarray:
        """The picture at its size: every pixel the mean of the `over` by `over` that were drawn for it."""
        return np.array(self.image.reduce(self.over) if self.over > 1 else self.image, dtype=np.uint8)

    def mirror(self) -> None:
        """Turns what is drawn so far left to right."""
        self.image = ImageOps.mirror(self.image)
        self.pen = ImageDraw.Draw(self.image)

    # -- shapes: each is left out when it has no size, so that a layout holds on a picture of 8 pixels --

    def box(self, left: float, top: float, right: float, bottom: float, fill: Color | None, corner: float = 0.0,
            outline: Color | None = None, width: float = 0.0) -> None:
        if right - left < 1 or bottom - top < 1:
            return
        half = min(right - left, bottom - top) / 2
        width = min(max(1, round(width)), int(half)) if outline else 0
        self.pen.rounded_rectangle((left, top, right - 1, bottom - 1), min(corner, half), fill=fill,
                                   outline=outline if width else None, width=width)

    def disc(self, x: float, y: float, radius: float, fill: Color | None, outline: Color | None = None,
             width: float = 0.0) -> None:
        if radius >= 0.5:
            self.pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline,
                             width=max(1, round(width)) if outline else 0)

    def line(self, points: Sequence[tuple[float, float]], fill: Color, width: float) -> None:
        self.pen.line([(round(x), round(y)) for x, y in points], fill=fill, width=max(1, round(width)))

    def stamp(self, mask: Image.Image, x: float, y: float, fill: Color) -> None:
        """A symbol in one colour, its square's top left corner here."""
        self.image.paste(tuple(fill), (round(x), round(y)), mask)

    # -- text: fit gives a line the size it may have, put sets it and counts it --

    def fit(self, words: str, cap: float, width: float, height: float = math.inf, bold: bool = False) -> _Line | None:
        """`words` on one line at the largest size that leaves its capitals no higher than `cap`, and its ink
        no wider than `width` and no higher than `height`. The picture bounds all three: no capital above
        SMALL_TYPE of it unless it is a label, no ink into the margin. None if no size is small enough."""
        cap = min(cap, self.most)
        width, height = min(width, self.w - 2 * self.edge), min(height, self.h - 2 * self.edge)
        if not words.strip() or cap < 1 or width < 1 or height < 1:      # nothing to set, or no room at all for it
            return None
        # A first size from the font's own measures, a tenth over what they allow, so that the letters are
        # drawn to be measured at about the size they end at and never at many times it.
        run = _typeface(self.font).getlength(words) / _typeface(self.font).size
        size = int(1.1 * min(min(cap, height) / _capital_share(self.font), width / max(run, 1e-6))) + 1
        while size >= 1:
            font, stroke = _font(self.font, size), (max(1, round(_BOLD * size)) if bold else 0)
            rise, sink = _capital(self.font, size, stroke)
            # The font's own box first, which costs no drawing and is no smaller than the ink; then the ink.
            left, top, right, bottom = font.getbbox(words, anchor="ls", stroke_width=stroke)
            excess = max((rise + sink) / cap, (right - left) / width, (bottom - top) / height)
            if excess <= 1.0:
                ink = _ink(font, words, stroke)
                if ink is None:                    # too small to leave a mark
                    return None
                if ink[2] - ink[0] <= width and ink[3] - ink[1] <= height:
                    return _Line(words, font, stroke, *ink, rise, sink)
            size = min(size - 1, int(size / max(excess, 1.0)))
        return None

    def put(self, line: _Line | None, x: float, baseline: float, fill: Color, align: str = "left") -> None:
        """Sets a line with the left end, the middle or the right end of its ink at `x`, moved as far as it
        takes to keep its ink out of the picture's margin."""
        if line is None:
            return
        x -= {"left": line.left, "centre": (line.left + line.right) / 2, "right": line.right}[align]
        x = min(max(x, self.edge - line.left), self.w - self.edge - line.right)
        baseline = min(max(baseline, self.edge - line.top), self.h - self.edge - line.bottom)
        self.pen.text((round(x), round(baseline)), line.words, font=line.font, fill=tuple(fill), anchor="ls",
                      stroke_width=line.stroke, stroke_fill=tuple(fill))
        self.heights.append(line.cap / self.h)

    def glyph(self, letter: str, x: float, y: float, cap: float, fill: Color) -> None:
        """One letter as a mark, its ink centred on a point. It is a picture, and no height is listed for it."""
        font = _font(self.font, max(1, round(cap / _capital_share(self.font))))
        ink = _ink(font, letter, 0)
        if ink is not None:
            self.pen.text((round(x - (ink[0] + ink[2]) / 2), round(y - (ink[1] + ink[3]) / 2)), letter, font=font,
                          fill=tuple(fill), anchor="ls")


# ---- symbols: each drawn on a square mask, in shares of its side, white where the symbol is ----

class _Pad:
    """A square to draw a symbol on. `on=False` cuts out of what is drawn already."""

    def __init__(self, side: float):
        self.side = max(1, round(side))
        self.mask = Image.new("L", (self.side, self.side), 0)
        self.pen = ImageDraw.Draw(self.mask)

    def box(self, left: float, top: float, right: float, bottom: float, corner: float = 0.0, on: bool = True) -> None:
        self.pen.rounded_rectangle([self.side * value for value in (left, top, right, bottom)],
                                   radius=corner * self.side, fill=255 * on)

    def disc(self, x: float, y: float, radius: float, on: bool = True) -> None:
        self.pen.ellipse([self.side * value for value in (x - radius, y - radius, x + radius, y + radius)],
                         fill=255 * on)

    def area(self, points: Sequence[tuple[float, float]], on: bool = True) -> None:
        self.pen.polygon([(self.side * x, self.side * y) for x, y in points], fill=255 * on)

    def path(self, points: Sequence[tuple[float, float]], width: float, on: bool = True) -> None:
        """A stroke with round ends and round bends."""
        self.pen.line([(self.side * x, self.side * y) for x, y in points], fill=255 * on,
                      width=max(1, round(width * self.side)))
        for x, y in points:
            self.disc(x, y, width / 2, on)

    def half(self, left: float, flat: float, right: float, depth: float, up: bool = False, on: bool = True) -> None:
        """Half an ellipse with its straight side along `flat`: this deep below it, or this high above it."""
        self.pen.pieslice([self.side * value for value in (left, flat - depth, right, flat + depth)],
                          *((180, 360) if up else (0, 180)), fill=255 * on)

    def leaf(self, base: tuple[float, float], tip: tuple[float, float], breadth: float, on: bool = True) -> None:
        """A leaf from its stalk end to its point, this broad at the middle."""
        (x0, y0), (x1, y1) = base, tip
        length = math.hypot(x1 - x0, y1 - y0)
        across = ((y0 - y1) / length, (x1 - x0) / length)
        spine = [(k / 24, math.sin(math.pi * k / 24) ** 0.8 * breadth / 2) for k in range(25)]
        sides = [[(x0 + t * (x1 - x0) + way * out * across[0], y0 + t * (y1 - y0) + way * out * across[1])
                  for t, out in spine] for way in (1, -1)]
        self.area(sides[0] + sides[1][::-1], on)

    def turn(self, degrees: float) -> None:
        """Turns what is drawn about the middle, anticlockwise."""
        self.mask = self.mask.rotate(degrees, resample=Image.Resampling.BICUBIC)
        self.pen = ImageDraw.Draw(self.mask)


def _information(pad: _Pad) -> None:
    """The lower-case i of an information point: a dot over a stem with a flag and a foot."""
    pad.disc(0.5, 0.17, 0.1)
    pad.box(0.35, 0.36, 0.59, 0.45, corner=0.012)
    pad.box(0.42, 0.36, 0.59, 0.88)
    pad.box(0.31, 0.8, 0.7, 0.9, corner=0.012)


def _ticket(pad: _Pad) -> None:
    """A ticket on the slant: a bite out of each end, a row of holes where its stub tears off, two lines of print."""
    pad.box(0.08, 0.27, 0.92, 0.73, corner=0.04)
    for x in (0.08, 0.92):
        pad.disc(x, 0.5, 0.08, on=False)
    for k in range(5):
        pad.box(0.67, 0.305 + 0.085 * k, 0.695, 0.355 + 0.085 * k, on=False)
    pad.box(0.21, 0.39, 0.57, 0.45, on=False)
    pad.box(0.21, 0.55, 0.47, 0.61, on=False)
    pad.turn(20.0)


def _printer(pad: _Pad) -> None:
    """A printer from the front: a sheet going in at the top, and one coming out with two lines on it."""
    pad.box(0.27, 0.1, 0.73, 0.4, corner=0.02)
    pad.box(0.32, 0.15, 0.68, 0.4, on=False)
    pad.box(0.08, 0.33, 0.92, 0.72, corner=0.07)
    pad.disc(0.8, 0.44, 0.032, on=False)
    pad.box(0.21, 0.56, 0.79, 0.73, on=False)
    pad.box(0.25, 0.58, 0.75, 0.9, corner=0.02)
    pad.box(0.3, 0.58, 0.7, 0.85, on=False)
    pad.box(0.36, 0.66, 0.64, 0.7)
    pad.box(0.36, 0.75, 0.56, 0.79)


def _cup(pad: _Pad) -> None:
    """A cup on a saucer with three threads of steam over it."""
    pad.disc(0.735, 0.52, 0.13)                    # the handle: a ring, of which the bowl hides the near half
    pad.disc(0.735, 0.52, 0.068, on=False)
    pad.half(0.18, 0.4, 0.74, 0.37)
    pad.box(0.18, 0.385, 0.74, 0.41)
    pad.box(0.12, 0.81, 0.8, 0.87, corner=0.03)
    for x in (0.33, 0.46, 0.59):
        pad.path([(x + 0.024 * math.sin(2 * math.pi * k / 12), 0.31 - 0.19 * k / 12) for k in range(13)], 0.036)


def _runner(pad: _Pad) -> None:
    """A figure running to the right: the trunk leaning into the run, one leg down ahead, one stretched out behind."""
    pad.disc(0.635, 0.17, 0.088)
    pad.path([(0.585, 0.345), (0.475, 0.56)], 0.15)
    pad.path([(0.6, 0.35), (0.735, 0.44), (0.845, 0.375)], 0.075)
    pad.path([(0.555, 0.345), (0.41, 0.385), (0.345, 0.5)], 0.075)
    pad.path([(0.49, 0.57), (0.655, 0.665), (0.62, 0.875), (0.715, 0.89)], 0.1)
    pad.path([(0.465, 0.58), (0.335, 0.73), (0.13, 0.765)], 0.1)


def _exit(pad: _Pad) -> None:
    """The figure running through a doorway, as the standard draws it: the doorway light, the figure dark where
    it is before the doorway and light where its back foot is still outside it."""
    door = _Pad(pad.side)
    door.box(0.27, 0.04, 0.95, 0.96)
    _runner(pad)
    pad.mask = ImageChops.difference(door.mask, pad.mask)


def _arrow(pad: _Pad) -> None:
    """An arrow to the right."""
    pad.area([(0.1, 0.4), (0.5, 0.4), (0.5, 0.2), (0.9, 0.5), (0.5, 0.8), (0.5, 0.6), (0.1, 0.6)])


def _flower(pad: _Pad) -> None:
    for k in range(6):
        pad.disc(0.5 + 0.25 * math.cos(math.pi * k / 3), 0.5 + 0.25 * math.sin(math.pi * k / 3), 0.16)
    pad.disc(0.5, 0.5, 0.13, on=False)
    pad.disc(0.5, 0.5, 0.08)


def _leaf(pad: _Pad) -> None:
    pad.leaf((0.2, 0.8), (0.86, 0.14), 0.5)
    pad.path([(0.1, 0.9), (0.62, 0.38)], 0.035, on=False)
    pad.path([(0.1, 0.9), (0.24, 0.76)], 0.05)


def _fruit(pad: _Pad) -> None:
    """An apple with a stalk and a leaf."""
    pad.disc(0.41, 0.6, 0.25)
    pad.disc(0.59, 0.6, 0.25)
    pad.disc(0.5, 0.33, 0.07, on=False)
    pad.path([(0.5, 0.4), (0.53, 0.2)], 0.045)
    pad.leaf((0.56, 0.3), (0.82, 0.13), 0.16)


def _sweet(pad: _Pad) -> None:
    """A sweet in its wrapper, twisted shut at both ends."""
    pad.area([(0.32, 0.5), (0.08, 0.32), (0.08, 0.68)])
    pad.area([(0.68, 0.5), (0.92, 0.32), (0.92, 0.68)])
    pad.disc(0.5, 0.5, 0.25, on=False)
    pad.disc(0.5, 0.5, 0.22)
    pad.path([(0.41, 0.6), (0.59, 0.4)], 0.05, on=False)
    pad.turn(18.0)


# The marks of a place whose trade has no symbol. Each is a plain figure that is no company's: no ring round a
# dot, no four squares, no pair of discs.
def _arch(pad: _Pad) -> None:
    pad.half(0.14, 0.66, 0.86, 0.44, up=True)
    pad.half(0.33, 0.66, 0.67, 0.22, up=True, on=False)
    pad.box(0.08, 0.72, 0.92, 0.83, corner=0.03)


def _bars(pad: _Pad) -> None:
    for k, top in enumerate((0.52, 0.32, 0.12)):
        pad.box(0.14 + 0.26 * k, top, 0.34 + 0.26 * k, 0.88, corner=0.04)


def _peaks(pad: _Pad) -> None:
    pad.area([(0.06, 0.84), (0.4, 0.2), (0.74, 0.84)])
    pad.area([(0.4, 0.84), (0.68, 0.4), (0.96, 0.84)], on=False)
    pad.area([(0.5, 0.84), (0.7, 0.52), (0.9, 0.84)])


def _dot(pad: _Pad) -> None:
    pad.disc(0.5, 0.5, 0.2)


def _brolly(pad: _Pad) -> None:
    """An umbrella left leaning: a canopy with a scalloped rim and a point, a stick, a crook for the hand."""
    pad.half(0.1, 0.5, 0.9, 0.38, up=True)
    for x in (0.233, 0.5, 0.767):
        pad.disc(x, 0.5, 0.133, on=False)
    pad.path([(0.5, 0.07), (0.5, 0.14)], 0.04)
    pad.path([(0.5, 0.3), (0.5, 0.82)], 0.045)
    pad.path([(0.57 - 0.07 * math.cos(math.pi * k / 8), 0.82 + 0.07 * math.sin(math.pi * k / 8)) for k in range(9)],
             0.045)
    pad.turn(-20.0)


def _brolly_panel(pad: _Pad) -> None:
    """The middle panel of that umbrella's canopy, which a picture of it prints in a second tone."""
    pad.half(0.367, 0.5, 0.633, 0.38, up=True)
    pad.disc(0.5, 0.5, 0.133, on=False)
    pad.turn(-20.0)


_SYMBOLS: dict[str, Callable[[_Pad], None]] = {
    "information": _information, "ticket": _ticket, "printer": _printer, "cup": _cup, "exit": _exit, "arrow": _arrow,
    "flower": _flower, "leaf": _leaf, "fruit": _fruit, "sweet": _sweet, "arch": _arch, "bars": _bars, "peaks": _peaks,
    "dot": _dot, "umbrella": _brolly, "umbrella panel": _brolly_panel}


def _symbol(name: str, side: float) -> Image.Image:
    """The mask of a symbol on a square of this side."""
    pad = _Pad(side)
    _SYMBOLS[name](pad)
    return pad.mask


# ---- the styles ----

def _label(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    line = sheet.fit(" ".join(art.text.split()), sheet.h, LABEL_WIDTH * sheet.w, LABEL_HEIGHT * sheet.h, bold=True)
    if line is not None:
        sheet.put(line, sheet.w / 2, line.baseline_of_ink(0, sheet.h), art.ink, "centre")


def _pictogram(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    name = _key(art.text)
    if name not in PICTOGRAMS:
        raise ValueError(f"unknown pictogram {art.text!r}: expected one of {', '.join(PICTOGRAMS)}")
    side = round(0.84 * sheet.unit)
    sheet.stamp(_symbol(name, side), (sheet.w - side) / 2, (sheet.h - side) / 2, art.ink)


def _mark(sheet: _Sheet, mark: str, lockup: str, x: float, y: float, side: float, art: SignArt, letter: str) -> None:
    """A logo's mark on the square at (x, y): the symbol in ink, or cut in the ground's colour out of a disc or
    a tile of ink. A monogram is the name's first letter cut out of one of the two."""
    if mark == "monogram" and lockup == "bare":
        lockup = "disc"
    if lockup == "bare":
        sheet.stamp(_symbol(mark, side), x, y, art.ink)
        return
    sheet.box(x, y, x + side, y + side, art.ink, corner=side / 2 if lockup == "disc" else 0.16 * side)
    if mark == "monogram":
        sheet.glyph(letter, x + side / 2, y + side / 2, 0.44 * side, art.ground)
        return
    inner = round((0.62 if lockup == "disc" else 0.7) * side)
    sheet.stamp(_symbol(mark, inner), x + (side - inner) / 2, y + (side - inner) / 2, art.ground)


def _logo(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    theme = _theme(art.text) or _PLAIN
    name = rng.choice(NAMES)                       # the first draw, so that the seed picks the name and no more
    mark, lockup = rng.choice(theme.marks), rng.choice(_LOCKUPS)
    trade = rng.choice(theme.trades) if theme.trades else ""
    w, h = sheet.w, sheet.h
    soft = _mix(art.ink, art.ground, 0.22)
    # Two ways to set it: the mark beside the name, or over it. The one that leaves the name larger is taken.
    side, gap, pad = round(0.64 * h), 0.2 * h, 0.14 * h
    beside = sheet.fit(name, 0.2 * h, w - 2 * pad - side - gap, bold=True)
    over_side = round(min(0.5 * h, 0.66 * w))
    over = sheet.fit(name, 0.12 * h, 0.86 * w, bold=True)
    if beside is not None and (over is None or beside.cap >= over.cap):
        small = sheet.fit(trade, 0.42 * beside.cap, max(beside.width, 0.3 * (w - 2 * pad - side - gap)))
        block = max(beside.width, small.width if small else 0.0)
        x = (w - side - gap - block) / 2
        _mark(sheet, mark, lockup, x, (h - side) / 2, side, art, name[0])
        stack = beside.cap + (0.55 * beside.cap + small.cap if small else 0.0)
        sheet.put(beside, x + side + gap, (h - stack) / 2 + beside.rise, art.ink)
        if small is not None:
            sheet.put(small, x + side + gap, (h + stack) / 2 - small.sink, soft)
        return
    small = sheet.fit(trade, 0.5 * over.cap, 0.86 * w) if over is not None else None
    stack = over_side + (0.09 * h + over.cap if over else 0.0) + (0.05 * h + small.cap if small else 0.0)
    y = (h - stack) / 2
    _mark(sheet, mark, lockup, (w - over_side) / 2, y, over_side, art, name[0])
    if over is not None:
        sheet.put(over, w / 2, y + over_side + 0.09 * h + over.rise, art.ink, "centre")
    if small is not None:
        sheet.put(small, w / 2, y + stack - small.sink, soft, "centre")


def _plate(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    theme = _theme(art.text)
    mark, word = (theme.marks[0] if theme else "dot"), _sentence(art.text)     # a word of no theme: a plain dot
    w, h, unit = sheet.w, sheet.h, sheet.unit
    sheet.box(0.05 * unit, 0.05 * unit, w - 0.05 * unit, h - 0.05 * unit, None, corner=0.07 * unit,
              outline=_mix(art.ground, art.ink, 0.3), width=0.012 * unit)             # a line inside its edge

    def tile(x: float, y: float, side: float) -> None:
        sheet.box(x, y, x + side, y + side, art.ink, corner=0.14 * side)
        inner = round(0.8 * side)
        sheet.stamp(_symbol(mark, inner), x + (side - inner) / 2, y + (side - inner) / 2, art.ground)

    if w >= 1.3 * h:                               # the pictogram at the left, the word beside it
        side, gap, pad = round(0.7 * h), 0.15 * h, 0.15 * h
        line = sheet.fit(word, 0.22 * h, w - 2 * pad - side - gap)
        x = (w - side - (gap + line.width if line else 0.0)) / 2
        tile(x, (h - side) / 2, side)
        sheet.put(line, x + side + gap, line.baseline(0, h) if line else 0.0, art.ink)
    else:                                          # the pictogram over the word
        side, gap = round(min(0.56 * w, 0.46 * h)), 0.09 * unit
        line = sheet.fit(word, 0.2 * unit, 0.76 * w)
        y = (h - side - (gap + line.cap if line else 0.0)) / 2
        tile((w - side) / 2, y, side)
        sheet.put(line, w / 2, y + side + gap + line.rise if line else 0.0, art.ink, "centre")


def _deal(lengths: Sequence[int], room: float, cost: float) -> list[int]:
    """How many rows of each section a column of a menu shows. The column has room for `room` rows, and a
    section's heading takes as much as `cost` rows; the sections have `lengths` rows to show.

    As many sections are begun as can show three rows each, or all they have; then the rows that are left are
    dealt to them in turn. A column too low for one such section shows what fits under the first heading."""
    counts: list[int] = []
    for length in lengths:
        least = min(3, length)
        if cost + least > room:
            break
        counts.append(least)
        room -= cost + least
    if not counts and lengths:
        return [max(0, min(lengths[0], int(room - cost)))]
    spare = int(room)
    while spare > 0 and any(count < length for count, length in zip(counts, lengths)):
        for k, length in enumerate(lengths[:len(counts)]):
            if spare > 0 and counts[k] < length:
                counts[k] += 1
                spare -= 1
    return counts


def _menu(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    theme = _theme(art.text)
    theme = theme if theme is not None and theme.sections else _PLAIN
    least, most, step = theme.prices
    # What it lists is drawn before it is laid out: the board's shape decides how much of it shows, not what it is.
    listing = [(title, [(item, f"{least + step * rng.randrange(round((most - least) / step) + 1):.2f}")
                        for item in rng.sample(items, len(items))]) for title, items in theme.sections]
    w, h, m = sheet.w, sheet.h, sheet.m
    pad = min(0.05 * m, 0.08 * sheet.unit)
    inner = w - 2 * pad
    item = min(_ITEM_M * m, 0.05 * h, inner / 15)            # a column holds a name and a price: 15 capitals wide
    head = min(_HEAD_M * m, 0.075 * h, 1.4 * item)
    columns = max(1, min(3, len(listing), int(inner / (19 * item))))
    gutter, pitch = 2.4 * item, 2.3 * item
    across = (inner - (columns - 1) * gutter) / columns
    heading, between = 2.3 * head, 0.7 * pitch               # a heading down to its first row; a section to the next
    # Each column takes every `columns`-th section, and shows of each what _deal gives it of the board's height.
    plan, stretch = [], 1.25
    for column in range(columns):
        mine = listing[column::columns]
        counts = _deal([len(rows) for _, rows in mine], (h - 2 * pad + between) / pitch, (heading + between) / pitch)
        shown = [(title, rows[:count]) for (title, rows), count in zip(mine, counts) if count]
        plan.append(shown)
        if shown:                                            # the rows may spread into what the column leaves free
            free = h - 2 * pad + between - len(shown) * (heading + between) - sum(counts) * pitch
            stretch = min(stretch, 1.0 + free / (sum(counts) * pitch))
    spread = pitch * max(1.0, stretch)
    names, prices = zip(*(row for shown in plan for _, rows in shown for row in rows)) if any(plan) else ((), ())
    run = _typeface(sheet.font).getlength
    dearest = sheet.fit(max(prices, key=run, default=""), item, 0.35 * across)
    room = across - (dearest.width if dearest else 0.0) - 1.6 * item
    longest = sheet.fit(max(names, key=run, default=""), dearest.cap if dearest else item, room)
    same = longest.cap if longest else 0.0
    soft = _mix(art.ground, art.ink, 0.45)
    for column, shown in enumerate(plan):
        x, y = pad + column * (across + gutter), pad
        for title, rows in shown:
            line = sheet.fit(title, head, across, bold=True)
            sheet.put(line, x, y + line.rise if line else 0.0, art.ink)
            sheet.box(x, y + 1.6 * head, x + across, y + 1.6 * head + max(1.0, 0.07 * head), soft)
            y += heading
            for name, price in rows:
                cost, line = sheet.fit(price, same, 0.35 * across), sheet.fit(name, same, room)
                baseline = y + (spread + item) / 2
                sheet.put(line, x, baseline, art.ink)
                sheet.put(cost, x + across, baseline, art.ink, "right")
                if line is not None and cost is not None:    # dots lead the eye from the name to its price
                    first, step = x + line.width + 0.7 * item, 0.55 * item
                    for k in range(int((across - cost.width - line.width - 1.4 * item) / step) + 1):
                        sheet.disc(first + step * k, baseline - 0.09 * item, 0.075 * item, soft)
                y += spread
            y += between


def _cuts(count: int, rng: random.Random) -> list[int]:
    """Which of `count` rows or columns of a plan are corridor: one, or two with rooms between them."""
    if count < 9:
        return [rng.randrange(2, count - 2)]
    return [rng.randrange(2, count // 2 - 1), rng.randrange(count // 2 + 1, count - 2)]


def _spans(count: int, cuts: list[int]) -> list[tuple[int, int]]:
    """The runs of rows or columns between the corridors."""
    edges = [-1, *cuts, count]
    return [(a + 1, b) for a, b in zip(edges, edges[1:]) if b > a + 1]


def _pieces(start: int, end: int, rng: random.Random) -> list[tuple[int, int]]:
    """A run of cells divided into rooms two to four cells long; what is left over joins the last."""
    rooms = []
    while start < end:
        length = min(end - start, rng.randint(2, 4))
        length += end - start - length == 1
        rooms.append((start, start + length))
        start += length
    return rooms


def _map(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    w, h, unit, m = sheet.w, sheet.h, sheet.unit, sheet.m
    ink, ground, marker = art.ink, art.ground, _marker(art.ink)
    pad = 0.05 * unit
    head = min(_ITEM_M * m, 0.042 * h)
    title = sheet.fit(rng.choice(_FLOORS), head, 0.6 * w, bold=True)
    sheet.put(title, pad, pad + title.rise if title else 0.0, ink)
    top = pad + 2.0 * head                         # by the size asked for, not the size set: the plan is the same
    type_px = min(0.02 * m, 0.03 * h)              # at every scale, and a pixel's rounding moves no wall of it
    swatch = 1.7 * type_px
    legend = ("You are here", "Stairs", "Lifts", "Toilets")
    at_side = w >= 1.3 * h                         # the legend down the right side, or in a row along the foot
    if at_side:
        legend_w = min(0.3 * w, swatch + 9.0 * type_px)
        left, right, bottom = pad, w - pad - legend_w - 0.6 * pad, h - pad
    else:
        left, right, bottom = pad, w - pad, h - pad - 2 * 2.5 * type_px - 0.4 * pad
    if right - left < 8 or bottom - top < 8:       # a sliver: no room for a plan under the title
        return

    # The plan is a grid of cells: one or two corridors each way, and the blocks between them divided into rooms.
    cell = math.sqrt((right - left) * (bottom - top) / 150)
    columns, rows = (min(24, max(5, round(span / cell))) for span in (right - left, bottom - top))
    cw, ch = (right - left) / columns, (bottom - top) / rows
    down, along = _cuts(columns, rng), _cuts(rows, rng)      # the corridors that run down the plan, and along it
    rooms = []
    for x0, x1 in _spans(columns, down):
        for y0, y1 in _spans(rows, along):
            if x1 - x0 >= y1 - y0:
                rooms += [(a, y0, b, y1) for a, b in _pieces(x0, x1, rng)]
            else:
                rooms += [(x0, a, x1, b) for a, b in _pieces(y0, y1, rng)]
    order = rng.sample(range(len(rooms)), len(rooms))
    kinds = {index: kind for index, kind in zip(order, ("stairs", "lifts", "toilets", "toilets"))}
    wall = max(1.0, 0.0045 * unit)

    def corner(column: float, row: float) -> tuple[float, float]:
        return left + column * cw, top + row * ch

    for index, (x0, y0, x1, y1) in enumerate(rooms):
        (a, b), (c, d) = corner(x0, y0), corner(x1, y1)
        kind, plain = kinds.get(index, "room"), rng.choice((0.1, 0.1, 0.19))     # most rooms pale, some a shade deeper
        tint = {"toilets": 0.42, "stairs": 0.0, "lifts": 0.06}.get(kind, plain)
        sheet.box(a - wall / 2, b - wall / 2, c + wall / 2, d + wall / 2, _mix(ground, ink, tint), outline=ink,
                  width=wall)
        _fitting(sheet, kind, a, b, c, d, ink, wall)
        # A door: a gap in the wall on a side that a corridor passes, part of the way along it.
        sides = [("left", x0 - 1 in down), ("right", x1 in down), ("top", y0 - 1 in along), ("bottom", y1 in along)]
        doors = [side for side, open_ in sides if open_]
        if doors:
            side = rng.choice(doors)
            if side in ("left", "right"):
                x, y = (a if side == "left" else c), b + (d - b) * rng.choice((0.3, 0.5, 0.7))
                sheet.box(x - wall, y - 0.22 * ch, x + wall, y + 0.22 * ch, ground)
            else:
                x, y = a + (c - a) * rng.choice((0.3, 0.5, 0.7)), (b if side == "top" else d)
                sheet.box(x - 0.22 * cw, y - wall, x + 0.22 * cw, y + wall, ground)
    sheet.box(left - wall, top - wall, right + wall, bottom + wall, None, outline=ink, width=2.2 * wall)
    # The way in: a corridor that runs down the plan is open at its foot. The marker stands just inside it.
    way_in = rng.choice(down)
    x, y = corner(way_in + 0.5, rows)
    sheet.box(x - 0.36 * cw, y - 2 * wall, x + 0.36 * cw, y + 2 * wall, ground)
    _here(sheet, *corner(way_in + 0.5, rows - 1.1), 0.38 * min(cw, ch), marker, ground)

    # The legend: down the side, or in two rows of two along the foot. Its entries are set in one size, the one
    # that the longest of them fits its room in, short of the next entry's swatch.
    room = (legend_w if at_side else (right - left) / 2) - 2.0 * swatch
    same = min((line.cap for line in (sheet.fit(name, type_px, room) for name in legend) if line), default=0.0)
    for k, name in enumerate(legend):
        if at_side:
            x, y = right + 0.6 * pad, top + k * 1.6 * swatch
        else:
            x, y = left + (k % 2) * (right - left) / 2, bottom + 0.5 * pad + (k // 2) * 2.5 * type_px
        if name == "You are here":
            _here(sheet, x + swatch / 2, y + swatch / 2, 0.36 * swatch, marker, ground)
        else:
            kind = name.lower()
            sheet.box(x, y, x + swatch, y + swatch, _mix(ground, ink, 0.42 if kind == "toilets" else 0.03),
                      outline=ink, width=wall)
            _fitting(sheet, kind, x, y, x + swatch, y + swatch, ink, wall)
        line = sheet.fit(name, same, room)
        sheet.put(line, x + 1.4 * swatch, line.baseline(y, y + swatch) if line else 0.0, ink)


def _here(sheet: _Sheet, x: float, y: float, radius: float, marker: Color, ground: Color) -> None:
    """The "you are here" marker: a dot of the marker's colour with a ring of the ground round it."""
    sheet.disc(x, y, 1.45 * radius, ground)
    sheet.disc(x, y, 1.45 * radius, None, outline=marker, width=0.22 * radius)
    sheet.disc(x, y, 0.85 * radius, marker)


def _fitting(sheet: _Sheet, kind: str, left: float, top: float, right: float, bottom: float, ink: Color,
             wall: float) -> None:
    """What a plan draws inside a room that is no plain room: the treads of stairs, the crossed car of a lift."""
    if kind == "stairs":                           # a flight along the room's longer side, its treads across it
        wide = right - left >= bottom - top
        length, depth = (right - left, bottom - top) if wide else (bottom - top, right - left)
        treads = max(2, round(length / (0.3 * depth + 1)))
        for k in range(1, treads):
            x, y = left + (right - left) * k / treads, top + (bottom - top) * k / treads
            sheet.line([(x, top), (x, bottom)] if wide else [(left, y), (right, y)], ink, 0.7 * wall)
    elif kind == "lifts":
        inset = 0.2 * min(right - left, bottom - top)
        sheet.box(left + inset, top + inset, right - inset, bottom - inset, None, outline=ink, width=0.8 * wall)
        sheet.line([(left + inset, top + inset), (right - inset, bottom - inset)], ink, 0.7 * wall)
        sheet.line([(left + inset, bottom - inset), (right - inset, top + inset)], ink, 0.7 * wall)


def _exit_sign(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    w, h, unit = sheet.w, sheet.h, sheet.unit
    sheet.box(0, 0, w, h, _EXIT_GREEN)
    side = round(0.86 * unit)
    if w >= 1.5 * h:                               # the arrow beside the doorway, on the side the figure runs to
        arrow = round(min(side, 0.9 * (w - 0.14 * unit) - side))
        x = (w - side - arrow) / 2
        sheet.stamp(_symbol("exit", side), x, (h - side) / 2, _EXIT_WHITE)
        sheet.stamp(_symbol("arrow", arrow), x + side, (h - arrow) / 2, _EXIT_WHITE)
    elif h >= 1.5 * w:                             # or under it
        arrow = round(min(side, 0.9 * (h - 0.14 * unit) - side))
        y = (h - side - arrow) / 2
        sheet.stamp(_symbol("exit", side), (w - side) / 2, y, _EXIT_WHITE)
        sheet.stamp(_symbol("arrow", arrow), (w - arrow) / 2, y + side, _EXIT_WHITE)
    else:                                          # a square sign is the doorway alone, as the standard's is
        sheet.stamp(_symbol("exit", side), (w - side) / 2, (h - side) / 2, _EXIT_WHITE)
    if art.seed % 2:                               # the same sign for a way out to the left
        sheet.mirror()


# ---- the pictures of a notice: each drawn on a block of the palette's palest tone, w across and h down ----
#
# A picture is a scene that fills its block from side to side (hills, bunting), or one thing (a figure, a clock,
# a warning, an umbrella). A thing has one measure, taken from the block's width and its height together, and
# every other measure of it is a share of that one: it is the same thing on a block of any shape.

def _hills(pen: ImageDraw.ImageDraw, w: int, h: int, tones: Sequence[Color], pale: Color, rng: random.Random) -> None:
    """Three ranges of hills under a sun."""
    sun = 0.11 * min(w, h) + 0.04 * h
    x, y = w * rng.uniform(0.2, 0.8), h * rng.uniform(0.25, 0.4)
    pen.ellipse((x - sun, y - sun, x + sun, y + sun), fill=pale)
    for tone, base, swell, start in ((tones[1], 0.62, 0.1, rng.random()), (tones[2], 0.74, 0.08, rng.random()),
                                     (tones[3], 0.88, 0.05, rng.random())):
        ridge = [(w * k / 40, h * (base + swell * math.sin(2 * math.pi * (start + 0.9 * k / 40)))) for k in range(41)]
        pen.polygon([*ridge, (w, h), (0, h)], fill=tone)


def _bunting(pen: ImageDraw.ImageDraw, w: int, h: int, tones: Sequence[Color], pale: Color, rng: random.Random) -> None:
    """Strings of flags from side to side, as over a fair: as many strings as the block is high for."""
    flag = 0.2 * min(w, h)
    strings, flags = max(1, min(4, round(h / (2.6 * flag)))), max(2, round(0.88 * w / flag))
    colours = (tones[2], pale, tones[3], tones[1])
    for k in range(strings):
        top, first = h * (k + 0.15) / strings, rng.randrange(len(colours))
        tilt, sag = rng.uniform(-0.4, 0.4) * flag, rng.uniform(0.5, 0.9) * flag

        def at(t: float) -> tuple[float, float]:   # the string, from beyond one side of the block to beyond the other
            return w * (1.1 * t - 0.05), top + tilt * (t - 0.5) + 4 * sag * t * (1 - t)

        for n in range(flags):
            (ax, ay), (bx, by) = at((n + 0.08) / flags), at((n + 0.92) / flags)
            pen.polygon([(ax, ay), (bx, by), ((ax + bx) / 2, (ay + by) / 2 + 1.15 * flag)],
                        fill=colours[(first + n) % len(colours)])
        pen.line([at(n / 40) for n in range(41)], fill=tones[3], width=max(1, round(0.06 * flag)))


def _figure(pen: ImageDraw.ImageDraw, w: int, h: int, tones: Sequence[Color], pale: Color, rng: random.Random) -> None:
    """A head and shoulders, standing on the foot of the block. Its measure is the head's radius."""
    r = min(w / 5.6, h / 5.0)
    pen.ellipse((w / 2 - 2.3 * r, h - 1.8 * r, w / 2 + 2.3 * r, h + 2.2 * r), fill=tones[3])
    pen.ellipse((w / 2 - r, h - 4.05 * r, w / 2 + r, h - 2.05 * r), fill=tones[2])


def _clock(pen: ImageDraw.ImageDraw, w: int, h: int, tones: Sequence[Color], pale: Color, rng: random.Random) -> None:
    """A clock in the middle of the block, at a time of the seed's. Its measure is its radius."""
    r, x, y = 0.38 * min(w, h), w / 2, h / 2
    hour, minute = rng.randrange(12), 5 * rng.randrange(12)

    def spoke(turn: float, near: float, far: float, width: float, fill: Color) -> None:
        across, down = r * math.sin(2 * math.pi * turn), -r * math.cos(2 * math.pi * turn)
        pen.line([(x + near * across, y + near * down), (x + far * across, y + far * down)], fill=fill,
                 width=max(1, round(width * r)))

    pen.ellipse((x - r, y - r, x + r, y + r), fill=tones[3])
    pen.ellipse((x - 0.84 * r, y - 0.84 * r, x + 0.84 * r, y + 0.84 * r), fill=pale)
    for k in range(12):
        spoke(k / 12, 0.6 if k % 3 == 0 else 0.68, 0.76, 0.05, tones[2])
    spoke((hour + minute / 60) / 12, -0.08, 0.4, 0.11, tones[3])
    spoke(minute / 60, -0.08, 0.64, 0.075, tones[3])
    pen.ellipse((x - 0.09 * r, y - 0.09 * r, x + 0.09 * r, y + 0.09 * r), fill=tones[3])


def _warning(pen: ImageDraw.ImageDraw, w: int, h: int, tones: Sequence[Color], pale: Color, rng: random.Random) -> None:
    """A warning triangle with its mark, in the middle of the block. Its measure is the reach of its corners."""
    s = 0.36 * min(w, h)
    x, y = w / 2, h / 2 + 0.25 * s

    def triangle(reach: float) -> list[tuple[float, float]]:
        thirds = (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
        return [(x + reach * math.sin(turn), y - reach * math.cos(turn)) for turn in thirds]

    outer = triangle(s)
    pen.polygon(outer, fill=tones[3])
    pen.line([*outer, *outer[:2]], fill=tones[3], width=max(1, round(0.3 * s)), joint="curve")    # its corners rounded
    pen.polygon(triangle(0.72 * s), fill=pale)
    pen.rounded_rectangle((x - 0.075 * s, y - 0.42 * s, x + 0.075 * s, y + 0.07 * s), radius=0.07 * s, fill=tones[3])
    pen.ellipse((x - 0.085 * s, y + 0.15 * s, x + 0.085 * s, y + 0.32 * s), fill=tones[3])


def _umbrella(pen: ImageDraw.ImageDraw, w: int, h: int, tones: Sequence[Color], pale: Color,
              rng: random.Random) -> None:
    """An umbrella in the middle of the block, as a thing that was left behind. Its measure is its square's side."""
    side = round(0.84 * min(w, h))
    for part, tone in (("umbrella", tones[3]), ("umbrella panel", tones[2])):
        pen.bitmap((round((w - side) / 2), round((h - side) / 2)), _symbol(part, side), fill=tone)


_PICTURES: dict[str, Callable[[ImageDraw.ImageDraw, int, int, Sequence[Color], Color, random.Random], None]] = {
    "hills": _hills, "bunting": _bunting, "figure": _figure, "clock": _clock, "warning": _warning,
    "umbrella": _umbrella}
assert all(kind in _PICTURES for headings in _HEADINGS.values() for _, kinds in headings for kind in kinds), \
    "a heading of a notice asks for a picture that is not drawn"


def _picture(width: float, height: float, ground: Color, kind: str, rng: random.Random) -> Image.Image:
    """A picture of this kind for a notice, in colours of its own that are dulled towards the sheet's as print is."""
    w, h = max(1, round(width)), max(1, round(height))
    image = Image.new("RGB", (w, h))
    pen = ImageDraw.Draw(image)
    hue = rng.random()
    # Four tones of neighbouring hues, from a pale one for what is far to a deep one for what is near, and one
    # from across the wheel for what catches the light.
    steps = ((0.0, 0.82, 0.5), (0.06, 0.64, 0.45), (0.11, 0.46, 0.45), (0.16, 0.3, 0.4))
    tones = [_mix(_shade(hue + turn, lightness, saturation), ground, 0.12) for turn, lightness, saturation in steps]
    pale = _mix(_shade(hue + 0.5, 0.88, 0.7), ground, 0.12)
    pen.rectangle((0, 0, w, h), fill=tones[0])
    _PICTURES[kind](pen, w, h, tones, pale, rng)
    return image


def _bars_of_text(sheet: _Sheet, left: float, top: float, right: float, bottom: float, bar: float, fill: Color,
                  rng: random.Random) -> None:
    """Paragraphs of small print as bars: full lines, and a short one where a paragraph ends."""
    y, pitch, to_go = top, 2.4 * bar, rng.randint(3, 5)
    while y + bar <= bottom:
        last = to_go == 1
        reach = rng.uniform(0.35, 0.7) if last else rng.uniform(0.9, 1.0)
        sheet.box(left, y, left + (right - left) * reach, y + bar, fill)
        y += pitch * (1.9 if last else 1.0)
        to_go = rng.randint(3, 5) if last else to_go - 1


def _notice(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    w, h, unit, m = sheet.w, sheet.h, sheet.unit, sheet.m
    headings = _HEADINGS.get(_key(art.text)) or (*_HEADINGS["events"], *_HEADINGS["notices"])
    (heading, kinds), when = rng.choice(headings), rng.choice(_WHEN)
    pad = 0.08 * unit
    bar = min(0.014 * h, 0.012 * m)
    grey = _mix(art.ground, art.ink, 0.36)
    if h >= 0.9 * w:                               # the picture across the top, the text under it
        left, top, right = pad, pad + 0.4 * (h - 2 * pad), w - pad
        box = (pad, pad, w - pad, top - 0.5 * pad)
        caption = None
    else:                                          # or the picture at the left, the text beside it
        left, top, right = pad + 0.42 * (w - 2 * pad), pad, w - pad
        box = (pad, pad, left - 0.6 * pad, h - pad)
        # A column much higher than it is wide is no frame for a picture: the picture stands at its head, a
        # little higher than wide, and the rest is its caption, if that leaves the caption two lines or more.
        caption = box[1] + _PICTURE_HIGH * (box[2] - box[0]) + 0.6 * pad
        if box[3] - caption >= 3.4 * bar:
            box = (*box[:3], caption - 0.6 * pad)
        else:
            caption = None
    if box[2] - box[0] >= 1 and box[3] - box[1] >= 1:
        sheet.image.paste(_picture(box[2] - box[0], box[3] - box[1], art.ground, rng.choice(kinds), rng),
                          (round(box[0]), round(box[1])))
    if caption is not None:
        _bars_of_text(sheet, box[0], caption, box[2], h - pad, bar, grey, rng)
    line = sheet.fit(heading, min(0.05 * h, _HEAD_M * m), right - left, bold=True)
    sheet.put(line, left, top + line.rise if line else 0.0, art.ink)
    y = top + 1.7 * (line.cap if line else 0.0)
    small = sheet.fit(when, 0.55 * line.cap if line else 0.0, right - left) if line else None
    sheet.put(small, left, y + small.rise if small else 0.0, _mix(art.ink, art.ground, 0.3))
    y += 2.2 * (small.cap if small else 0.0)
    foot = h - pad - 3.2 * bar
    _bars_of_text(sheet, left, y, right, foot - 2.4 * bar, bar, grey, rng)
    sheet.box(left, foot, right, foot + max(1.0, 0.25 * bar), _mix(art.ground, art.ink, 0.5))      # a rule, and under
    sheet.box(left, foot + 1.6 * bar, left + 0.4 * (right - left), foot + 2.6 * bar, grey)         # it, who to ask


def _screen(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    w, h, unit = sheet.w, sheet.h, sheet.unit
    ink, ground = art.ink, art.ground
    title, clock = rng.choice(_SCREEN_TITLES), f"{rng.randint(8, 19)}:{rng.randrange(60):02d}"
    prompt, status = rng.choice(_SCREEN_PROMPTS), rng.choice(_SCREEN_STATUS)
    buttons = [_sentence(art.text) or "Start", *rng.sample(_SCREEN_BUTTONS, 2)]
    soft, pad = _mix(ground, ink, 0.6), 0.07 * unit
    # The title bar: a square for the service's own mark, its name, and the time at the far end.
    bar = 0.16 * h
    sheet.box(0, 0, w, bar, _mix(ground, ink, 0.16))
    sheet.box(pad, 0.28 * bar, pad + 0.44 * bar, 0.72 * bar, ink, corner=0.1 * bar)
    time = sheet.fit(clock, 0.3 * bar, 0.25 * w)
    line = sheet.fit(title, 0.34 * bar, w - 2 * pad - 0.7 * bar - (time.width if time else 0.0) - pad)
    if line is not None and time is not None and time.cap > line.cap:        # on a narrow screen: no larger than it
        time = sheet.fit(clock, line.cap, 0.25 * w)
    sheet.put(time, w - pad, time.baseline(0, bar) if time else 0.0, soft, "right")
    sheet.put(line, pad + 0.7 * bar, line.baseline(0, bar) if line else 0.0, ink)
    line = sheet.fit(prompt, 0.065 * h, w - 2 * pad)
    sheet.put(line, pad, line.baseline(0.2 * h, 0.38 * h) if line else 0.0, soft)
    # The buttons: the first filled, the others in outline; in a row on a wide screen, in a column on any other.
    top, bottom, gap = 0.42 * h, 0.8 * h, 0.5 * pad
    if w >= 1.25 * h:
        widths = (0.44, 0.28, 0.28)
        room = w - 2 * pad - 2 * gap
        lefts = [pad + sum(widths[:k]) * room + k * gap for k in range(3)]
        boxes = [(lefts[k], top, lefts[k] + widths[k] * room, bottom) for k in range(3)]
    else:
        each = (bottom - top - 2 * gap) / 3
        boxes = [(pad, top + k * (each + gap), w - pad, top + k * (each + gap) + each) for k in range(3)]
    lines = [sheet.fit(word, (0.11 if k == 0 else 0.075) * h, 0.8 * (c - a), 0.6 * (d - b), bold=k == 0)
             for k, (word, (a, b, c, d)) in enumerate(zip(buttons, boxes))]
    same = min((line.cap for line in lines[1:] if line), default=0.0)        # the buttons in outline, in one size
    for k, (word, (a, b, c, d)) in enumerate(zip(buttons, boxes)):
        corner = 0.16 * min(c - a, d - b)
        sheet.box(a, b, c, d, ink if k == 0 else None, corner=corner, outline=None if k == 0 else soft,
                  width=0.012 * unit)
        line = lines[k] if k == 0 else sheet.fit(word, same, 0.8 * (c - a), 0.6 * (d - b))
        sheet.put(line, (a + c) / 2, line.baseline(b, d) if line else 0.0, ground if k == 0 else ink, "centre")
    # The status line: a lamp, what the machine is ready for, and which of its pages this is.
    line = sheet.fit(status, 0.045 * h, w - 2 * pad - 0.07 * h - 0.2 * h)     # short of the three dots
    middle = 0.9 * h
    sheet.disc(pad + 0.02 * h, middle, 0.018 * h, _LAMP_GREEN)
    sheet.put(line, pad + 0.07 * h, line.baseline(middle, middle) if line else 0.0, soft)
    for k in range(3):
        sheet.disc(w - pad - 0.012 * h - 0.05 * h * k, middle, 0.012 * h, ink if k == 2 else _mix(ground, ink, 0.35))


def _paper(sheet: _Sheet, art: SignArt, rng: random.Random) -> None:
    kind = rng.choice(("blank", "ruled", "margin"))
    if kind == "blank":
        return
    # any closer, the sheet is no longer the ground's colour but a grey of it.
    pitch = max(10, round(0.008 * sheet.m / sheet.over)) * sheet.over
    head = 2 * pitch + (round(0.1 * sheet.h) // sheet.over) * sheet.over
    faint = _mix(art.ground, art.ink, 0.16)
    for y in range(head, sheet.h - pitch // 2, pitch):
        sheet.box(0, y, sheet.w, y + sheet.over, faint)
    if kind == "margin":
        x = (round(0.14 * sheet.w) // sheet.over) * sheet.over
        sheet.box(x, 0, x + sheet.over, sheet.h, _mix(art.ground, _marker(art.ink), 0.3))


_STYLES: dict[str, Callable[[_Sheet, SignArt, random.Random], None]] = {
    "label": _label, "logo": _logo, "pictogram": _pictogram, "plate": _plate, "menu": _menu, "map": _map,
    "exit": _exit_sign, "notice": _notice, "screen": _screen, "paper": _paper}
assert set(_STYLES) == set(SIGN_STYLES), "a style of the scene description has no drawing, or a drawing no style"
