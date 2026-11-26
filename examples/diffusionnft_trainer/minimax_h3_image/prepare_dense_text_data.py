# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the dense-text OCR prompt set used by ``run_minimax_h3_image_ocr_lora.sh``.

Writes ``train_<tier>.parquet`` (5760 rows) and ``test_<tier>.parquet`` (256 rows) for every length tier. Rows are
user-turn-only chat prompts with ``reward_model.ground_truth`` set to the target text and ``extra_info.split`` /
``extra_info.index`` set for the reward's prompt lookup. Test texts never occur in train, and a quarter of the test
rows use six scenes that never occur in train. Fully deterministic.

    python prepare_dense_text_data.py --output-dir ~/data/minimax_h3_image_dense_text
"""

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20261110
TRAIN_PER_TIER, TEST_PER_TIER = 5760, 256
TIERS = {"t3": (2, 4, 1), "t8": (6, 10, 2), "t20": (16, 24, 4), "t40": (34, 46, 8)}

ADJ = (
    "fresh warm quiet golden silver bright early late local small grand open daily simple modern classic "
    "gentle bold clear deep plain crisp sweet smoky royal rapid steady hidden lucky rustic urban coastal "
    "northern southern eastern western frozen roasted handmade organic vintage digital public private central "
    "annual weekly final first second third green purple orange scarlet ivory amber copper marble wooden "
    "woolen cotton velvet silent lively humble clever famous secret narrow broad distant nearby seasonal"
).split()
NOUN = (
    "bread coffee market garden river harbor station library museum theater bakery kitchen window lantern "
    "bridge tower valley meadow forest island canyon desert summit village avenue corner gallery studio "
    "workshop festival concert lecture seminar journey voyage picnic harvest winter summer spring autumn "
    "morning evening sunrise midnight ticket voucher receipt invoice parcel letter notice bulletin journal "
    "chapter volume recipe pepper ginger lemon cherry walnut almond butter honey cheese noodle dumpling "
    "pancake biscuit orchard vineyard cottage cabin castle chapel pavilion fountain stadium airport ferry "
    "tram bicycle engine compass anchor ladder hammer candle mirror blanket pillow jacket sweater scarf "
    "violin guitar trumpet piano drum painter baker tailor sailor farmer doctor teacher pilot captain "
    "member visitor neighbor student driver"
).split()
VERB = (
    "open close bring carry visit enjoy share serve bake brew build paint plant water gather collect deliver "
    "repair return follow watch listen learn teach travel explore discover remember celebrate welcome join "
    "meet greet invite reserve order taste sample choose compare measure count record print sign send keep"
).split()
PREP = "near behind beside under above across beyond inside outside before after toward around through".split()
NUM = "two three four five six seven eight nine ten twelve twenty thirty forty fifty sixty ninety".split()

SCENES = [
    (
        "a photorealistic close-up of a cafe chalkboard menu leaning against a red brick wall, lit by warm afternoon "
        "light, with a few coffee cups softly blurred in the background",
        "Hand-lettered on the board in white chalk",
    ),
    (
        "a clean studio photograph of a concert poster pinned to a cork wall, flat even lighting, matte paper with a "
        "pale cream background and a thin dark border",
        "Printed on the poster in dark ink",
    ),
    (
        "a realistic street photograph of a metal notice plate bolted to a concrete wall beside a doorway, overcast "
        "daylight, slight weathering on the plate edges",
        "Engraved on the plate in black paint-filled letters",
    ),
    (
        "a macro product photograph of a glass jar with a white paper label, standing on a wooden kitchen counter, "
        "soft window light from the left",
        "Printed on the label in small black type",
    ),
    (
        "an overhead photograph of an open hardcover book lying on a linen tablecloth, the left page blank and the "
        "right page carrying a short printed passage, natural daylight",
        "Printed on the right page in a serif typeface",
    ),
    (
        "a photograph of a classroom whiteboard mounted on a pale green wall, fluorescent lighting, a tray of markers "
        "below it",
        "Written on the whiteboard in blue marker",
    ),
    (
        "a museum interior photograph of a small rectangular placard mounted beside a display case, warm spotlights, "
        "shallow depth of field",
        "Printed on the placard in dark grey type",
    ),
    (
        "a straight-on photograph of a smartphone lying on a dark desk, its screen showing a plain white note-taking "
        "app, the room dim around it",
        "Displayed on the screen in black sans-serif type",
    ),
    (
        "a photograph of a wooden shop sign hanging from an iron bracket above a narrow cobbled street, early morning "
        "light, shallow depth of field",
        "Painted on the sign in cream-colored letters",
    ),
    (
        "a realistic photograph of a yellow warning board fixed to a chain-link fence at a construction site, bright "
        "midday sun, dust in the air",
        "Printed on the board in bold black letters",
    ),
    (
        "a flat-lay photograph of a greeting card on a pastel pink surface, surrounded by a few dried flowers and a "
        "ribbon, soft diffuse light",
        "Written inside the card in neat dark blue handwriting",
    ),
    (
        "a close-up photograph of a paper train ticket held between two fingers, a blurred platform behind it, cool "
        "morning light",
        "Printed on the ticket in small black type",
    ),
    (
        "an overhead photograph of a recipe card resting on a flour-dusted marble worktop next to a wooden spoon, "
        "bright kitchen lighting",
        "Typed on the card in a typewriter font",
    ),
    (
        "a wide photograph of a roadside billboard against a clear blue sky, seen from the road below, the billboard "
        "face plain white",
        "Printed across the billboard in large dark letters",
    ),
    (
        "a studio photograph of a plain white cotton t-shirt laid flat on a light wooden floor, even soft lighting",
        "Screen-printed on the chest in black letters",
    ),
    (
        "a night photograph of a shop window with a glowing sign box mounted inside, reflections of street lights on "
        "the glass",
        "Shown on the sign box in bright white letters on a dark background",
    ),
    (
        "a close-up photograph of a yellow sticky note stuck to the edge of a computer monitor, office lighting, a "
        "keyboard blurred below",
        "Handwritten on the note in black pen",
    ),
    (
        "a product photograph of a brown cardboard shipping box on a conveyor belt in a warehouse, overhead industrial "
        "lighting",
        "Stamped on the side of the box in dark ink",
    ),
    (
        "a photograph of a wall calendar page hanging from a nail on a white plaster wall, morning light from a nearby "
        "window casting a soft shadow",
        "Printed on the page in dark red type",
    ),
    (
        "a photograph of a green road sign on a steel pole beside a country highway, fields stretching behind it, late "
        "afternoon sun",
        "Printed on the sign in white reflective letters",
    ),
    (
        "a photograph of a community bulletin board in a library hallway, a single sheet of white paper pinned in its "
        "center, soft indoor lighting",
        "Printed on the sheet in black type",
    ),
    (
        "a straight-on photograph of a laptop on a wooden cafe table, the screen showing a plain presentation slide "
        "with a light grey background",
        "Displayed on the slide in dark sans-serif type",
    ),
    (
        "a close-up photograph of a brass memorial plaque set into a stone wall in a park, dappled light through "
        "leaves",
        "Engraved on the plaque in dark recessed letters",
    ),
    (
        "a photograph of a folded newspaper lying on a park bench, its front page facing up, slightly creased, bright "
        "daylight",
        "Printed on the front page in black newsprint type",
    ),
]
N_TRAIN_SCENES = 18
ORDINAL = ("1", "2", "3", "4", "5", "6", "7", "8", "9")


def _words(rng: random.Random, n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        kind = rng.randrange(6)
        if kind == 0:
            chunk = [rng.choice(ADJ), rng.choice(NOUN)]
        elif kind == 1:
            chunk = [rng.choice(VERB), "the", rng.choice(NOUN)]
        elif kind == 2:
            chunk = [rng.choice(PREP), "the", rng.choice(ADJ), rng.choice(NOUN)]
        elif kind == 3:
            chunk = [rng.choice(NUM), rng.choice(ADJ), rng.choice(NOUN) + "s"]
        elif kind == 4:
            chunk = [rng.choice(NOUN), "and", rng.choice(NOUN)]
        else:
            chunk = [rng.choice(VERB), rng.choice(ADJ), rng.choice(NOUN) + "s"]
        out.extend(chunk)
    out = out[:n]
    while out[-1] in ("the", "and") or out[-1] in PREP:
        out[-1] = rng.choice(NOUN)
    return out


def make_text(rng: random.Random, tier: str) -> list[str]:
    low, high, n_lines = TIERS[tier]
    n = rng.randint(low, high)
    words = _words(rng, n)
    base, extra = divmod(n, n_lines)
    lines, pos = [], 0
    for index in range(n_lines):
        size = base + (1 if index < extra else 0)
        line = words[pos : pos + size]
        pos += size
        line[0] = line[0].capitalize()
        lines.append(" ".join(line))
    return lines


def make_prompt(scene: tuple[str, str], lines: list[str]) -> str:
    description, medium = scene
    if len(lines) == 1:
        text = f'{medium} is a single short line of small text that reads "{lines[0]}".'
    else:
        parts = " ".join(f'Line {ORDINAL[i]} reads "{line}".' for i, line in enumerate(lines))
        text = (
            f"{medium} is a compact block of small text set in exactly {len(lines)} lines, "
            f"stacked from top to bottom. {parts}"
        )
    return (
        f"The image is {description}. {text} The lettering is small, crisp and fully legible, evenly spaced, "
        "and every word is spelled exactly as given. There is no other text anywhere in the image."
    )


def dense_rows(tier: str, split: str, count: int, taken: set[str]) -> list[dict]:
    rng = random.Random(f"{SEED}-{tier}-{split}")
    n_scenes = N_TRAIN_SCENES if split == "train" else len(SCENES)
    rows = []
    while len(rows) < count:
        lines = make_text(rng, tier)
        target = "\n".join(lines)
        if target in taken:
            continue
        taken.add(target)
        scene_id = len(rows) % n_scenes
        if split == "test":
            scene_id = (len(rows) * 7) % len(SCENES)
        index = len(rows)
        rows.append(
            {
                "data_source": f"ocr_dense_{tier}",
                "prompt": [{"role": "user", "content": make_prompt(SCENES[scene_id], lines)}],
                "ability": "ocr",
                "reward_model": {"style": "model", "ground_truth": target},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "tier": tier,
                    "n_words": sum(len(line.split()) for line in lines),
                    "n_chars": sum(len(line.replace(" ", "")) for line in lines),
                    "n_lines": len(lines),
                    "lines": lines,
                    "scene_id": scene_id,
                    "scene_in_train": scene_id < N_TRAIN_SCENES,
                    "target_text": target,
                },
            }
        )
    return rows


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    taken: set[str] = set()
    for tier in TIERS:
        test = dense_rows(tier, "test", TEST_PER_TIER, taken)
        train = dense_rows(tier, "train", TRAIN_PER_TIER, taken)
        write(train, args.output_dir / f"train_{tier}.parquet")
        write(test, args.output_dir / f"test_{tier}.parquet")
        print(f"{tier}: {len(train)} train / {len(test)} test -> {args.output_dir}")


if __name__ == "__main__":
    main()
