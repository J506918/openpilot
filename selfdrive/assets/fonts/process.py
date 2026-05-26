#!/usr/bin/env python3
from pathlib import Path
import json

import pyray as rl

FONT_DIR = Path(__file__).resolve().parent
SELFDRIVE_DIR = FONT_DIR.parents[1]
TRANSLATIONS_DIR = SELFDRIVE_DIR / "ui" / "translations"
LANGUAGES_FILE = TRANSLATIONS_DIR / "languages.json"

GLYPH_PADDING = 6
EXTRA_CHARS = "–‑✓×°§•X⚙✕◀▶✔⌫⇧␣●↳çêüñ€£¥"

# Per-language source font and size
# MiSans @ 48 for high-DPI, unifont @ 16 for ko/th (bitmap only)
_LANG_FONT = {
    "en":    ("MiSans-Demibold.otf", 48),
    "de":    ("MiSans-Demibold.otf", 48),
    "fr":    ("MiSans-Demibold.otf", 48),
    "pt-BR": ("MiSans-Demibold.otf", 48),
    "es":    ("MiSans-Demibold.otf", 48),
    "tr":    ("MiSans-Demibold.otf", 48),
    "uk":    ("MiSans-Demibold.otf", 48),
    "zh-CHS": ("MiSans-Demibold.otf", 48),
    "zh-CHT": ("MiSans-Demibold.otf", 48),
    "ja":    ("MiSans-Demibold.otf", 48),
    "ko":    ("unifont.otf", 16),
    "th":    ("unifont.otf", 16),
}


def _languages() -> dict[str, str]:
    """Return {native_name: lang_code} from languages.json."""
    if not LANGUAGES_FILE.exists():
        return {}
    with LANGUAGES_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def _lang_char_sets() -> dict[str, tuple[int, ...]]:
    """Return {lang_code: sorted_codepoint_tuple} for each language.

    Each set: base ASCII + EXTRA_CHARS + that language's PO file characters.
    Does NOT include all_lang_names — those go only in menu.fnt.
    """
    base_ascii = set(map(chr, range(32, 127))) | set(EXTRA_CHARS)

    result: dict[str, set[str]] = {}
    for lang_code in _LANG_FONT:
        chars = set(base_ascii)
        po_path = TRANSLATIONS_DIR / f"app_{lang_code}.po"
        if po_path.exists():
            chars.update(po_path.read_text(encoding="utf-8"))
        result[lang_code] = tuple(sorted(ord(c) for c in chars))

    return result


def _menu_codepoints() -> tuple[int, ...]:
    """Codepoints for the language menu — unifont 16px only."""
    languages = _languages()
    base_ascii = set(map(chr, range(32, 127))) | set(EXTRA_CHARS)
    all_lang_names = set("".join(languages.keys()))
    return tuple(sorted(ord(c) for c in (base_ascii | all_lang_names)))


def _glyph_metrics(glyphs, rects, codepoints):
    entries = []
    min_offset_y, max_extent = None, 0
    for idx, codepoint in enumerate(codepoints):
        glyph = glyphs[idx]
        rect = rects[idx]
        width = int(round(rect.width))
        height = int(round(rect.height))
        offset_y = int(round(glyph.offsetY))
        min_offset_y = offset_y if min_offset_y is None else min(min_offset_y, offset_y)
        max_extent = max(max_extent, offset_y + height)
        entries.append({
            "id": codepoint,
            "x": int(round(rect.x)),
            "y": int(round(rect.y)),
            "width": width,
            "height": height,
            "xoffset": int(round(glyph.offsetX)),
            "yoffset": offset_y,
            "xadvance": int(round(glyph.advanceX)),
        })

    if min_offset_y is None:
        raise RuntimeError("No glyphs were generated")

    line_height = int(round(max_extent - min_offset_y))
    base = int(round(max_extent))
    return entries, line_height, base


def _write_bmfont(path: Path, font_size: int, face: str, atlas_name: str,
                  line_height: int, base: int, atlas_size, entries):
    if line_height != font_size:
        print("using font size for line height", atlas_name)
        line_height = font_size
    lines = [
        f'info face="{face}" size=-{font_size} bold=0 italic=0 charset="" unicode=1 stretchH=100 smooth=0 aa=1 padding=0,0,0,0 spacing=0,0 outline=0',
        f"common lineHeight={line_height} base={base} scaleW={atlas_size[0]} scaleH={atlas_size[1]} pages=1 packed=0 alphaChnl=0 redChnl=4 greenChnl=4 blueChnl=4",
        f'page id=0 file="{atlas_name}"',
        f"chars count={len(entries)}",
    ]
    for entry in entries:
        lines.append(
            ("char id={id:<4} x={x:<5} y={y:<5} width={width:<5} height={height:<5} " +
             "xoffset={xoffset:<5} yoffset={yoffset:<5} xadvance={xadvance:<5} page=0  chnl=15").format(**entry)
        )
    path.write_text("\n".join(lines) + "\n")


def _process(font_file: str, font_size: int, stem: str, codepoints: tuple[int, ...]):
    font_path = FONT_DIR / font_file
    if not font_path.exists():
        raise FileNotFoundError(f"Source font not found: {font_path}")

    print(f"Generating {stem} from {font_file} @ {font_size}px ({len(codepoints)} codepoints)...")

    data = font_path.read_bytes()
    file_buf = rl.ffi.new("unsigned char[]", data)
    cp_buffer = rl.ffi.new("int[]", codepoints)
    cp_ptr = rl.ffi.cast("int *", cp_buffer)
    glyphs = rl.load_font_data(
        rl.ffi.cast("unsigned char *", file_buf), len(data),
        font_size, cp_ptr, len(codepoints), rl.FontType.FONT_DEFAULT)
    if glyphs == rl.ffi.NULL:
        raise RuntimeError("raylib failed to load font data")

    rects_ptr = rl.ffi.new("Rectangle **")
    image = rl.gen_image_font_atlas(glyphs, rects_ptr, len(codepoints),
                                     font_size, GLYPH_PADDING, 0)
    if image.width == 0 or image.height == 0:
        raise RuntimeError("raylib returned an empty atlas")

    rects = rects_ptr[0]
    atlas_name = f"{stem}.png"
    atlas_path = FONT_DIR / atlas_name
    entries, line_height, base = _glyph_metrics(glyphs, rects, codepoints)

    if not rl.export_image(image, atlas_path.as_posix()):
        raise RuntimeError("Failed to export atlas image")

    _write_bmfont(FONT_DIR / f"{stem}.fnt", font_size, stem,
                  atlas_name, line_height, base, (image.width, image.height), entries)
    print(f"  → {stem}.fnt + {stem}.png ({image.width}×{image.height})")


def main():
    # 1. Per-language atlases
    lang_sets = _lang_char_sets()
    for lang_code in sorted(lang_sets):
        font_file, font_size = _LANG_FONT[lang_code]
        _process(font_file, font_size, lang_code, lang_sets[lang_code])

    # 2. Menu atlas (unifont 16px, all language names)
    _process("unifont.otf", 16, "menu", _menu_codepoints())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
