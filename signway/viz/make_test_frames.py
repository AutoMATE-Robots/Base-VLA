"""Paste a drawn 'Elevators' plate with a GRAPHIC right arrow (OCR cannot read it, so the
gate must ask the VLM) onto a camera frame, at 4 growing sizes. Usage: base.png out_dir"""
import sys
from PIL import Image, ImageDraw, ImageFont
base = Image.open(sys.argv[1]).convert("RGB"); out = sys.argv[2]
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
for i, letter_px in enumerate([22, 38, 60, 85]):
    im = base.copy(); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font_path, int(letter_px * 1.35))
    tw = d.textlength("Elevators", font=f); pad = letter_px * 0.5; aw = letter_px * 2.2
    w, h = tw + aw + 3 * pad, letter_px * 1.35 + 2 * pad
    x0, y0 = 1250 - w / 2, 380 - h / 2
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(18, 52, 110))
    d.text((x0 + pad, y0 + pad * 0.6), "Elevators", font=f, fill="white")
    ax, ay, s = x0 + 2 * pad + tw, y0 + h / 2, letter_px
    d.polygon([(ax, ay - .25*s), (ax + 1.2*s, ay - .25*s), (ax + 1.2*s, ay - .6*s), (ax + 2*s, ay),
               (ax + 1.2*s, ay + .6*s), (ax + 1.2*s, ay + .25*s), (ax, ay + .25*s)], fill="white")
    im.save(f"{out}/f{i}.png")
print("ok")
