"""Render docs/demo.gif as a typed-out replay of real `formulaforge demo` output.

Usage: python scripts/make_demo_gif.py demo_output.txt docs/demo.gif [font.ttf]
The input is the captured stdout of `python -m formulaforge demo --table sales` fed with the requests below;
frames are drawn with Pillow (no screen recording), total duration ~30 s.
"""
import sys

from PIL import Image, ImageDraw, ImageFont

BG, FG, DIM, GREEN, YELLOW, RED = (24, 26, 31), (220, 223, 228), (130, 136, 148), (120, 220, 140), (240, 200, 100), (240, 110, 110)
W, H, LINES, FS = 860, 620, 30, 16


def main(src, dst, font_path="DejaVuSansMono.ttf"):
    raw = open(src).read().splitlines()
    table = raw[: raw.index("Type a spreadsheet request (empty line to quit).") ]
    table = [l for l in table if l.strip()]
    outs = [l for l in raw if l.startswith(("> formula", "formula")) or l.startswith("result")]
    pairs = [(outs[i].replace("> ", ""), outs[i + 1]) for i in range(0, len(outs) - 1, 2)]
    requests = [("What is the total Revenue?", True), ("Count rows where Region is East", True),
                ("Average Quantity where Product is Apples", True), ("Total Revenue where Region is West", False)]
    font = ImageFont.truetype(font_path, FS)
    frames, durs = [], []

    def render(lines):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((12, 8), "python -m formulaforge demo --table sales   (replay of real output)", font=font, fill=DIM)
        for i, (text, color) in enumerate(lines[-LINES:]):
            d.text((12, 36 + i * 18), text, font=font, fill=color)
        return im.convert("P", palette=Image.ADAPTIVE, colors=32)

    def add(lines, dur):
        frames.append(render(lines))
        durs.append(dur)

    screen = [(l, FG) for l in table]
    add(screen, 3000)
    for (req, ok), (formula, result) in zip(requests, pairs):
        screen.append(("", FG))
        for n in range(0, len(req) + 1, 3):  # typing effect
            add(screen + [("> " + req[:n] + "_", FG)], 120)
        screen.append(("> " + req, FG))
        add(screen, 700)
        screen.append((formula, YELLOW))
        add(screen, 1500)
        screen.append((result, GREEN if ok else RED))
        if not ok:
            screen.append(("  ^ wrong column: asked for Revenue (C), got Quantity (D)", RED))
        add(screen, 3500)
    add(screen, 3000)
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print(f"{dst}: {len(frames)} frames, {sum(durs) / 1000:.1f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
