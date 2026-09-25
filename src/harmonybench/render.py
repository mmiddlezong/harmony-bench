"""Render MusicXML to SVG with Verovio, then to PNG with resvg."""

import resvg_py
import verovio

_OPTIONS = {
    "adjustPageHeight": True,
    "adjustPageWidth": True,
    "pageMarginTop": 60,
    "pageMarginBottom": 60,
    "pageMarginLeft": 60,
    "pageMarginRight": 60,
    "scale": 60,
    "header": "none",
    "footer": "none",
    "breaks": "none",
}


def musicxml_to_svg(xml: str) -> str:
    tk = verovio.toolkit()
    tk.setOptions(_OPTIONS)
    if not tk.loadData(xml):
        raise ValueError("Verovio could not load MusicXML")
    return tk.renderToSVG(1)


def svg_to_png(svg: str, zoom: float = 2.0) -> bytes:
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", zoom=zoom))
