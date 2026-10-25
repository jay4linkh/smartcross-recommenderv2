"""
================================================================================
 SmartCross — explainer video builder
================================================================================

 Builds video/SmartCross_explainer.mp4: designed slides, a spoken voice-over
 (Windows' built-in speech voice, offline) and matching subtitles. Each
 narrated sentence gets its own frame, so the subtitle and the point being
 revealed on screen always line up with the voice.

     # with the console running (python web/backend/app.py), refresh screenshots:
     bash video/capture_app.sh
     # then, with the project's Python:
     .venv/Scripts/python.exe video/make_video.py

 Every number in the narration comes from defense/outputs/defense_results.json,
 the trained model, or the project docs. Change a sentence in SCENES and
 rerun to rebuild the whole video.
================================================================================
"""

import json
import os
import subprocess
import sys
import wave

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
ASSETS = os.path.join(HERE, "assets")
BUILD = os.path.join(HERE, "build")
OUT = os.path.join(HERE, "SmartCross_explainer.mp4")
VOICE = "Microsoft Zira Desktop"      # or "Microsoft David Desktop"
RATE = 1                               # -10 (slow) .. 10 (fast)
GAP = 0.35                             # silence after each sentence, seconds
SCENE_GAP = 0.5                        # extra silence at the end of a scene

W, H = 1920, 1080
C = dict(bg=(20, 8, 43), bg2=(40, 18, 80), card=(38, 18, 76), card2=(52, 26, 100),
         line=(88, 50, 160), violet=(109, 40, 217), bright=(139, 92, 246), lav=(196, 181, 253),
         text=(239, 233, 255), dim=(184, 169, 224), mute=(140, 125, 185), mint=(52, 211, 153),
         rose=(248, 113, 113), amber=(251, 191, 36), white=(255, 255, 255), sky=(96, 165, 250))

FONTS = {"r": "segoeui.ttf", "b": "segoeuib.ttf", "s": "seguisb.ttf", "m": "consola.ttf", "mb": "consolab.ttf"}
_fc = {}


def F(size, w="r"):
    k = (size, w)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(os.path.join("C:\\Windows\\Fonts", FONTS[w]), size)
    return _fc[k]


# ============================================================= live numbers
def load_numbers():
    E = json.load(open(os.path.join(ROOT, "defense", "outputs", "defense_results.json")))
    eng = E["engines"]
    return {
        "pop": eng["pop"]["top3"], "cf_old": eng["cf_old"]["top3"], "cf": eng["cf"]["top3"],
        "content": eng["content"]["top3"], "blend_old": eng["blend_old"]["top3"],
        "fused": eng["fused"]["top3"], "pipe": eng["pipe"]["top3"],
        "f1": eng["fused"]["macro_f1"], "removed": E["pipeline"]["truth_removed_by_rule"],
        "nh": eng["fused_nh"]["top3"], "nh_top1": eng["fused_nh"]["top1"], "nh_f1": eng["fused_nh"]["macro_f1"],
        "top1": eng["fused"]["top1"],
        "tilt": E["pipeline"]["top1_change_from_tilt"], "ece": E["calibration"]["ece"],
        "test": E["split"]["test_events"], "train": E["split"]["train_events"],
    }


def cf_example(holdings):
    """What the served collaborative engine says for one portfolio."""
    sys.path.insert(0, os.path.join(ROOT, "web", "backend"))
    import joblib
    import numpy as np
    import peers
    B = joblib.load(os.path.join(ROOT, "web", "backend", "model", "smartcross.pkl"))
    names, prods = B["all_names"], B["products"]
    v = np.zeros((1, len(names)), dtype=np.float32)
    for h in holdings:
        v[0, names.index(h)] = 1
    p = peers.pattern(B["peers"], v)[0]
    held = set(holdings)
    order = [j for j in np.argsort(-p) if prods[j] not in held][:4]
    return [(prods[j].replace("_", " "), float(p[j])) for j in order]


N = load_numbers()
pc = lambda v: f"{v*100:.1f}%"
CF_HOLD = ["current_account", "payroll_account", "direct_debit"]
CF_EX = cf_example(CF_HOLD)

# ================================================================== scenes
# Each scene: kind, chapter, title, data, and steps of (spoken sentence, reveal).
SCENES = [
    dict(kind="title", chapter="", title="", data={}, steps=[
        ("This is SmartCross: a recommender that tells a bank adviser which three products to offer a customer next, and why.", 1),
        ("In the next few minutes: the data, the attributes the model reads, how the two engines work, and how well it really performs.", 2),
    ]),
    dict(kind="bullets", chapter="01 · THE PROBLEM", title="Offer the right product, not every product", data={"items": [
        ("Same list for everyone", "Banks push one generic product list, and most customers ignore it."),
        ("Existing customers are the cheapest growth", "Selling a second product costs far less than winning a new customer."),
        ("Our goal", "For each customer: rank the products they are most likely to take next, keep only eligible ones, explain each."),
    ]}, steps=[
        ("Banks usually push the same product list to everyone, and most customers ignore it.", 1),
        ("Yet selling a second product to an existing customer is far cheaper than winning a new one.", 2),
        ("So the goal is simple: for each customer, rank the products they are most likely to take next, keep only the ones they are eligible for, and explain each one.", 3),
    ]),
    dict(kind="cards", chapter="02 · THE DATA", title="Three sources of data", data={"cards": [
        ("Santander bank", "2.2 GB", ["13.6 million rows", "24 products, month by month", "Jan 2015 – May 2016", "the main dataset"], "violet"),
        ("LendingClub", "1.6 GB", ["2.26 million US loans", "credit attributes", "Part A only:", "the first experiment"], "mute"),
        ("Market prices", "1 MB", ["9 price series", "sector funds, bank stocks", "monthly", "reads the economy"], "sky"),
    ]}, steps=[
        ("The project uses three sources of data.", 0),
        ("First, the Santander bank dataset from Kaggle: 2.2 gigabytes and 13.6 million rows, recording which of 24 products each customer held, month by month, from January 2015 to May 2016.", 1),
        ("Second, LendingClub: 1.6 gigabytes of US loans. We used it only in Part A, as a first experiment.", 2),
        ("Third, nine market price series, such as sector funds and bank stocks, which tell us whether the economy is booming or under stress.", 3),
    ]),
    dict(kind="bullets", chapter="02 · THE DATA", title="Part A: why LendingClub was not enough", data={"items": [
        ("The question", "Can credit data predict which loan a person wants?"),
        ("The answer: no", "Income, credit score and debt describe how much someone can borrow, not what for. Best model: 1.7 times chance."),
        ("The lesson", "The problem needs each customer's product history over time. Santander has it."),
    ]}, steps=[
        ("Part A asked whether credit data alone can predict which loan a person wants.", 1),
        ("It can't. Income, credit score and debt describe how much someone can borrow, not what they want it for. Even the best model reached only 1.7 times chance.", 2),
        ("That failure is why we moved to Santander, which has what LendingClub lacks: each customer's product history over time.", 3),
    ]),
    dict(kind="events", chapter="03 · FROM RAW DATA TO EVENTS", title="Every 'no → yes' is a decision we can learn from", data={}, steps=[
        ("Here is how the Santander data becomes training data.", 0),
        ("Each customer has one row per month, with a yes or no flag for every product.", 1),
        ("When a flag flips from no to yes, that customer has just taken a new product.", 2),
        (f"Each flip becomes one training example: what they held the month before, who they are, and the product they added. That gives 563,310 acquisition events across 22 products.", 3),
    ]),
    dict(kind="attributes", chapter="04 · THE ATTRIBUTES", title="35 attributes + 3 months of history", data={}, steps=[
        ("For every event, the model reads 35 attributes.", 0),
        ("24 of them are simply the products the customer already holds, such as a current account, a payroll account, a credit card or a pension.", 1),
        ("The other 11 describe the person and their behaviour: age, income, how long they have been with the bank, their segment, whether they are active, how many products they hold, and how recently they last added one.", 2),
        ("The target is the product they actually added next: one of 22 classes.", 3),
        ("On top of that, the model sees each customer's recent history: what they held two months ago, so it can spot a product they just dropped, and how many months in a row they have kept each product. That is 48 more features.", 4),
    ]),
    dict(kind="timeline", chapter="05 · TRAIN AND TEST", title="Split by time, never at random", data={}, steps=[
        ("We split the data by time, never at random, so the model can never peek at the future.", 0),
        (f"It trains on February 2015 to February 2016: {N['train']:,} events.", 1),
        ("February 2016 also serves as a validation month, used only to choose settings.", 2),
        (f"And it is tested exactly once, on March to May 2016: {N['test']:,} events it has never seen.", 3),
    ]),
    dict(kind="flow", chapter="06 · THE PIPELINE", title="Four layers, always in this order", data={}, steps=[
        ("The system is a pipeline of four layers.", 0),
        ("First, the customer becomes a profile: 35 numbers, plus three months of history.", 1),
        ("Second, two engines score all 22 products.", 2),
        ("Third, the economy nudges the scores.", 3),
        ("Fourth, eligibility rules remove anything the customer cannot have.", 4),
        ("What is left is the top three, each with a reason.", 5),
    ]),
    dict(kind="cf", chapter="07 · ENGINE 1 — COLLABORATIVE", title="What did customers like you add next?", data={}, steps=[
        ("The first engine is collaborative filtering. It ignores who the customer is, and asks: what did other customers with exactly this portfolio add next?", 1),
        (f"For example, among customers holding a current account, a payroll account and direct debit, the most common next products were {CF_EX[0][0]} and {CF_EX[1][0]}.", 2),
        ("If a portfolio is rare, the engine leans back on simpler single-product patterns, so it never trusts a handful of people.", 3),
        (f"Our first version ranked by lift, which rewards surprising but rare products. It reached only {N['cf_old']*100:.0f} percent top three. Ranking by probability instead raised it to {N['cf']*100:.0f} percent.", 4),
    ]),
    dict(kind="fusion", chapter="08 · ENGINE 2 — CONTENT, FUSED", title="Fusion, not averaging", data={}, steps=[
        ("The second engine is the content engine: a gradient boosting model, an ensemble of small decision trees.", 1),
        ("It reads the 35 profile attributes, the 48 history features, and the 44 scores from the collaborative engine: 127 inputs in total.", 2),
        ("This is called feature fusion. Instead of averaging the two engines, the model learns, for each customer, how much to trust the peers.", 3),
        ("That matters, because our first hybrid simply averaged them fifty-fifty, and it was measurably worse than the content engine on its own.", 4),
    ]),
    dict(kind="macro", chapter="09 · LAYER 3 — THE ECONOMY", title="The economy gets a vote", data={}, steps=[
        ("The third layer reads the economy.", 0),
        ("Nine price series each vote on the market's momentum and volatility, and the majority sets one of four regimes: expansion, recovery, slowdown or stress.", 1),
        ("Each regime gently tilts product scores. In stress, safe products like current accounts rise, and investment funds fall.", 2),
        (f"It changes the top recommendation for only {N['tilt']*100:.1f} percent of customers. It is there for business sense, not accuracy, and we say so.", 3),
    ]),
    dict(kind="rules", chapter="10 · LAYER 4 — ELIGIBILITY", title="Rules remove, never down-weight", data={}, steps=[
        ("The fourth layer is eligibility.", 0),
        ("Thirteen readable rules. For example, a mortgage needs age 25 to 65, an income of at least 30,000, and two years with the bank.", 1),
        ("A product that fails a rule is removed, not just pushed down, and missing data never counts as a pass.", 2),
        (f"This has a cost: the rules remove the product the customer actually took {N['removed']*100:.1f} percent of the time, and we report that openly.", 3),
    ]),
    dict(kind="image", chapter="11 · ONE REAL CUSTOMER", title="A real customer, scored live", data={"img": "app_story_engine.png"}, steps=[
        ("Here is a real customer from the test months, drawn at random by the app.", 1),
        ("The bars show every product's score at each stage: what peers did, what the content engine concluded, and the economy's nudge, with the regime's boosts marked in yellow.", 1),
        ("At the end, the app checks the recommendation against what this customer actually did next.", 1),
    ]),
    dict(kind="image", chapter="12 · THE APP", title="Why this answer, in plain words", data={"img": "app_why.png"}, steps=[
        ("The app shows this for any customer: what we recommend and why, what was removed and by which rule, and what else was considered.", 1),
        ("An adviser can also slide between the content engine and the raw peer opinion, and watch the ranking change live.", 1),
    ]),
    dict(kind="image", chapter="12 · THE APP", title="Inside the engines", data={"img": "app_engines.png"}, steps=[
        ("The Inside the Engines view shows the whole architecture, with a measured number on every box, and exactly which months were used for training and testing.", 1),
    ]),
    dict(kind="bars", chapter="13 · RESULTS", title="Right product in our top three?", data={}, steps=[
        (f"Now the results, on the {N['test']:,} test events. The question is: was the product they actually took among our three suggestions?", 1),
        (f"Picking three products at random would succeed about 14 percent of the time. Always recommending the most popular products gets {N['pop']*100:.1f} percent.", 2),
        (f"The improved collaborative engine alone reaches {N['cf']*100:.1f} percent, and the content engine {N['content']*100:.1f}.", 3),
        (f"Fusing the two, without history, reached {N['nh']*100:.1f} percent.", 4),
        (f"Adding three months of holdings history raises it to {N['fused']*100:.1f} percent, the best of all. After the economy and the rules, the full system still gets {N['pipe']*100:.1f} percent.", 5),
        (f"Its macro F1 score, which weighs rare products as much as common ones, is {N['f1']:.3f}: almost eight times the chance level.", 6),
    ]),
    dict(kind="compare", chapter="13 · RESULTS", title="What three months of history added", data={}, steps=[
        ("Here is what the history features added, on the same test customers.", 1),
        (f"The first suggestion is now right {N['top1']*100:.1f} percent of the time, up from {N['nh_top1']*100:.1f}.", 2),
        (f"And macro F1, the fairness score across rare products, rose from {N['nh_f1']:.3f} to {N['f1']:.3f}: a gain of about 28 percent.", 3),
    ]),
    dict(kind="image", chapter="14 · IS IT REAL?", title="Is the improvement real?", data={"img": "app_proof.png"}, steps=[
        ("Are these gains real, or luck? We resampled the test customers a thousand times, and compared every engine on identical samples.", 1),
        ("Every gain from the history features is real: the confidence interval excludes zero for top one, top three and macro F1.", 1),
        ("We also tried a hyperparameter search, per-product thresholds and XGBoost on the GPU. Each was judged on the validation month, none beat the chosen model, and all three are reported.", 1),
        ("Results are stable across all three test months, and the model is well calibrated: when it says 60 percent, it is right about 60 percent of the time.", 1),
    ]),
    dict(kind="summary", chapter="SUMMARY", title="SmartCross in one slide", data={"items": [
        ("Two engines, fused, plus history", "Collaborative + content + 3 months of holdings, from 563,310 decisions"),
        ("Economy + eligibility on top", "Every recommendation checked and explained"),
        (f"{pc(N['pipe'])} end-to-end top three", "vs about 14% random and 70.5% for the popularity list"),
    ]}, steps=[
        ("To sum up.", 0),
        ("Two engines, fused rather than averaged, reading three months of each customer's history, learned from 563 thousand real product decisions.", 1),
        ("An economy layer and hard eligibility rules on top, with every recommendation explained.", 2),
        (f"And {N['pipe']*100:.1f} percent of the time, the product the customer actually took next is in our top three. Thank you.", 3),
    ]),
]


# ================================================================ drawing
def gradient():
    img = Image.new("RGB", (W, H), C["bg"])
    px = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        col = tuple(int(C["bg"][i] * (1 - t) + C["bg2"][i] * t) for i in range(3))
        px.line([(0, y), (W, y)], fill=col)
    return img


BASE = None


def logo(d, x, y, s):
    d.rounded_rectangle([x, y, x + s, y + s], radius=s // 4, fill=C["violet"])
    pts = [(x + s * .22, y + s * .70), (x + s * .42, y + s * .42), (x + s * .58, y + s * .56), (x + s * .80, y + s * .28)]
    d.line(pts, fill=C["white"], width=max(3, s // 12), joint="curve")
    r = s * .08
    d.ellipse([pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r], fill=C["white"])


def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.getlength(t) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def frame(scene_i, scene, reveal, sentence):
    img = BASE.copy()
    d = ImageDraw.Draw(img, "RGBA")
    if scene["kind"] != "title":
        logo(d, 70, 48, 48)
        d.text((132, 50), "SmartCross", font=F(30, "b"), fill=C["white"])
        d.text((W - 70, 58), scene["chapter"], font=F(22, "mb"), fill=C["lav"], anchor="ra")
        d.text((110, 140), scene["title"], font=F(58, "b"), fill=C["white"])
    RENDER[scene["kind"]](img, d, scene["data"], reveal)
    # subtitle
    f = F(34, "s")
    lines = wrap(sentence, f, W - 360)
    if len(lines) > 2:
        f = F(29, "s")
        lines = wrap(sentence, f, W - 300)
    lh = f.size + 12
    bh = lh * len(lines) + 34
    y0 = 1046 - bh
    d.rounded_rectangle([140, y0, W - 140, y0 + bh], radius=18, fill=(10, 4, 24, 222))
    for i, ln in enumerate(lines):
        d.text((W / 2, y0 + 17 + i * lh), ln, font=f, fill=C["white"], anchor="ma")
    # progress
    d.rectangle([0, H - 6, W, H], fill=(255, 255, 255, 25))
    d.rectangle([0, H - 6, int(W * (scene_i + 1) / len(SCENES)), H], fill=C["bright"])
    return img


def a(col, on):
    """colour at full or ghost strength"""
    return col + ((255,) if on else (40,))


def r_title(img, d, data, rv):
    logo(d, W // 2 - 70, 230, 140)
    d.text((W / 2, 420), "SmartCross", font=F(120, "b"), fill=C["white"], anchor="ma")
    d.text((W / 2, 575), "Know what to offer before they ask.", font=F(46, "s"), fill=C["lav"], anchor="ma")
    if rv >= 2:
        d.text((W / 2, 670), "the data  ·  the attributes  ·  two engines  ·  the results",
               font=F(30, "m"), fill=C["dim"], anchor="ma")


def r_bullets(img, d, data, rv):
    y = 270
    for i, (h, b) in enumerate(data["items"]):
        on = i < rv
        d.rounded_rectangle([110, y, W - 110, y + 170], radius=22, fill=a(C["card"], on),
                            outline=a(C["line"], on), width=2)
        d.ellipse([150, y + 50, 220, y + 120], fill=a(C["violet"], on))
        d.text((185, y + 85), str(i + 1), font=F(36, "b"), fill=a(C["white"], on), anchor="mm")
        d.text((260, y + 30), h, font=F(40, "b"), fill=a(C["white"], on))
        for k, ln in enumerate(wrap(b, F(30), W - 480)[:2]):
            d.text((260, y + 85 + k * 38), ln, font=F(30), fill=a(C["dim"], on))
        y += 195


def r_cards(img, d, data, rv):
    n, x0, gap = len(data["cards"]), 110, 40
    cw = (W - 2 * x0 - gap * (n - 1)) / n
    for i, (name, big, lines, col) in enumerate(data["cards"]):
        on = i < rv
        x = x0 + i * (cw + gap)
        d.rounded_rectangle([x, 270, x + cw, 860], radius=26, fill=a(C["card"], on), outline=a(C[col], on), width=3)
        d.text((x + 40, 305), name, font=F(42, "b"), fill=a(C["white"], on))
        d.text((x + 40, 380), big, font=F(96, "b"), fill=a(C[col] if col != "mute" else C["lav"], on))
        for k, ln in enumerate(lines):
            d.text((x + 40, 540 + k * 60), ("• " if k < 3 else "") + ln, font=F(32, "s" if k == 3 else "r"),
                   fill=a(C["amber"] if k == 3 else C["dim"], on))


def r_events(img, d, data, rv):
    months = ["Sep 2015", "Oct 2015", "Nov 2015", "Dec 2015"]
    rows = [("current account", [1, 1, 1, 1]), ("payroll account", [0, 1, 1, 1]),
            ("pensions", [0, 0, 0, 1]), ("credit card", [0, 0, 0, 0])]
    x0, y0, cw, rh = 140, 300, 170, 90
    on = rv >= 1
    d.text((x0, y0 - 55), "customer #48,213  ·  one row per month", font=F(28, "m"), fill=a(C["lav"], on))
    for j, m in enumerate(months):
        d.text((x0 + 330 + j * cw + cw / 2, y0 + 20), m, font=F(28, "s"), fill=a(C["dim"], on), anchor="ma")
    for i, (name, vals) in enumerate(rows):
        y = y0 + 80 + i * rh
        d.text((x0, y + 22), name, font=F(32, "s"), fill=a(C["white"], on))
        for j, v in enumerate(vals):
            cx = x0 + 330 + j * cw + cw / 2
            flip = rv >= 2 and name == "pensions" and j == 3
            fill = C["mint"] if flip else (C["violet"] if v else C["card2"])
            d.rounded_rectangle([cx - 55, y + 10, cx + 55, y + 70], radius=14, fill=a(fill, on))
            d.text((cx, y + 40), "yes" if v else "no", font=F(28, "b"), fill=a(C["white"], on), anchor="mm")
    if rv >= 2:
        cx = x0 + 330 + 3 * cw + cw / 2
        y = y0 + 80 + 2 * rh + 40
        d.ellipse([cx - 72, y - 42, cx + 72, y + 42], outline=C["mint"], width=5)
        yb = y0 + 80 + 4 * rh + 20
        d.text((cx + 60, yb), "no → yes  =  a new product", font=F(30, "b"), fill=C["mint"], anchor="ra")
    if rv >= 3:
        x = 1210
        d.rounded_rectangle([x, 290, W - 110, 860], radius=26, fill=C["card"], outline=C["mint"], width=3)
        d.text((x + 40, 320), "ONE TRAINING EXAMPLE", font=F(26, "mb"), fill=C["mint"])
        items = [("held last month", "current account, payroll account"), ("who they are", "age 41 · income 52,000 · 4 yrs"),
                 ("→ added", "PENSIONS")]
        for k, (lab, val) in enumerate(items):
            d.text((x + 40, 390 + k * 115), lab, font=F(26), fill=C["dim"])
            d.text((x + 40, 425 + k * 115), val, font=F(34 if k < 2 else 44, "b"), fill=C["mint"] if k == 2 else C["white"])
        d.text((x + 40, 760), "563,310 events · 22 products", font=F(34, "b"), fill=C["amber"])


PRODUCTS24 = ["savings", "guarantees", "current", "derivatives", "payroll acct", "junior", "más particular",
              "particular", "particular plus", "short deposits", "medium deposits", "long deposits", "e-account",
              "funds", "mortgage", "pension plan", "loans", "taxes", "credit card", "securities", "home account",
              "payroll", "pensions", "direct debit"]
PERSON11 = [("age", ""), ("income", "-1 = not on file"), ("seniority", "months with the bank"),
            ("segment", "VIP / individual / student"), ("active", "yes / no"), ("new customer", "yes / no"),
            ("gender", ""), ("products held", "count"), ("months observed", ""), ("months since last product", ""),
            ("products added so far", "")]


def r_attributes(img, d, data, rv):
    on1, on2 = rv >= 1, rv >= 2
    d.rounded_rectangle([110, 250, 1000, 815], radius=24, fill=a(C["card"], on1), outline=a(C["violet"], on1), width=3)
    d.text((150, 285), "24  products held  (yes / no)", font=F(36, "b"), fill=a(C["white"], on1))
    for k, p in enumerate(PRODUCTS24):
        col, row = k % 3, k // 3
        x, y = 150 + col * 280, 335 + row * 58
        d.rounded_rectangle([x, y, x + 260, y + 48], radius=24, fill=a(C["card2"], on1))
        d.text((x + 130, y + 24), p, font=F(24, "s"), fill=a(C["lav"], on1), anchor="mm")
    d.rounded_rectangle([1040, 250, W - 110, 815], radius=24, fill=a(C["card"], on2), outline=a(C["sky"], on2), width=3)
    d.text((1080, 285), "11  about the person", font=F(36, "b"), fill=a(C["white"], on2))
    for k, (p, note) in enumerate(PERSON11):
        y = 335 + k * 43
        d.text((1080, y), "• " + p, font=F(29, "s"), fill=a(C["white"], on2))
        if note:
            d.text((1460, y + 3), note, font=F(24), fill=a(C["dim"], on2))
    if rv == 3:
        d.text((W / 2, 832), "target  →  the product added next  (22 classes)", font=F(36, "b"),
               fill=C["amber"], anchor="ma")
    if rv >= 4:
        d.text((W / 2, 832), "+ 48 history:  held 2 months ago  ·  months held in a row", font=F(36, "b"),
               fill=C["mint"], anchor="ma")


def r_timeline(img, d, data, rv):
    months = ["F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D", "J", "F", "M", "A", "M"]
    years = {0: "2015", 11: "2016"}
    x0, cw, y = 140, 102, 420
    for i, m in enumerate(months):
        if i <= 11:
            col, on = C["violet"], rv >= 1
        elif i == 12:
            col, on = C["amber"], rv >= 2
        else:
            col, on = C["mint"], rv >= 3
        if i == 12 and rv == 1:
            col, on = C["violet"], True
        d.rounded_rectangle([x0 + i * cw + 4, y, x0 + (i + 1) * cw - 4, y + 120], radius=14, fill=a(col, on))
        d.text((x0 + i * cw + cw / 2, y + 60), m, font=F(36, "b"), fill=a(C["white"], on), anchor="mm")
        if i in years:
            d.text((x0 + i * cw + 6, y - 50), years[i], font=F(30, "m"), fill=C["dim"])
    if rv >= 1:
        d.text((x0, y + 160), f"TRAIN  ·  Feb 2015 – Feb 2016  ·  {N['train']:,} events", font=F(36, "b"), fill=C["lav"])
    if rv >= 2:
        d.text((x0, y + 220), "VALIDATION  ·  Feb 2016  ·  used only to choose settings", font=F(36, "b"), fill=C["amber"])
    if rv >= 3:
        d.text((x0, y + 280), f"TEST  ·  Mar – May 2016  ·  {N['test']:,} events never seen", font=F(36, "b"), fill=C["mint"])


def box(d, xy, title, sub, col, on=True, tsize=34):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=22, fill=a(C["card"], on), outline=a(C[col], on), width=4)
    d.text(((x0 + x1) / 2, y0 + (y1 - y0) / 2 - 22), title, font=F(tsize, "b"), fill=a(C["white"], on), anchor="mm")
    for k, ln in enumerate(sub.split("\n")):
        d.text(((x0 + x1) / 2, y0 + (y1 - y0) / 2 + 22 + k * 32), ln, font=F(26), fill=a(C["dim"], on), anchor="mm")


def arrow(d, x0, y0, x1, y1, on=True, col="lav"):
    d.line([(x0, y0), (x1, y1)], fill=a(C[col], on), width=5)
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (-0.5, 0.5):
        d.line([(x1, y1), (x1 - 22 * math.cos(ang + s), y1 - 22 * math.sin(ang + s))], fill=a(C[col], on), width=5)


def r_flow(img, d, data, rv):
    items = [("Profile", "35 numbers
+ 48 history", "lav"), ("Two engines", "collaborative +\ncontent, fused", "violet"),
             ("Economy", "regime tilts\nthe scores", "sky"), ("Rules", "remove what they\ncannot have", "rose"),
             ("Top 3", "each with\na reason", "mint")]
    bw, gap, y0 = 290, 62, 420
    x = 110
    for i, (t, s, col) in enumerate(items):
        on = i < rv
        box(d, (x, y0, x + bw, y0 + 230), t, s, col, on, 40)
        if i < len(items) - 1:
            arrow(d, x + bw + 6, y0 + 115, x + bw + gap - 6, y0 + 115, i + 1 < rv)
        x += bw + gap
    d.text((110, y0 + 290), "layer 1", font=F(26, "m"), fill=C["mute"])
    d.text((110 + (bw + gap), y0 + 290), "layer 2", font=F(26, "m"), fill=C["mute"])
    d.text((110 + 2 * (bw + gap), y0 + 290), "layer 3", font=F(26, "m"), fill=C["mute"])
    d.text((110 + 3 * (bw + gap), y0 + 290), "layer 4", font=F(26, "m"), fill=C["mute"])
    d.text((110 + 4 * (bw + gap), y0 + 290), "output", font=F(26, "m"), fill=C["mute"])


def r_cf(img, d, data, rv):
    d.rounded_rectangle([110, 260, 860, 860], radius=26, fill=C["card"], outline=C["violet"], width=3)
    d.text((150, 290), "THIS PORTFOLIO", font=F(26, "mb"), fill=C["lav"])
    for k, h in enumerate(CF_HOLD):
        d.rounded_rectangle([150, 345 + k * 75, 820, 405 + k * 75], radius=30, fill=C["violet"])
        d.text((485, 375 + k * 75), h.replace("_", " "), font=F(30, "b"), fill=C["white"], anchor="mm")
    d.text((150, 600), "customers with exactly these", font=F(30), fill=C["dim"])
    d.text((150, 640), "products — what did they add next?", font=F(30), fill=C["dim"])
    if rv >= 3:
        d.text((150, 730), "rare portfolio? → lean on single-", font=F(28, "s"), fill=C["amber"])
        d.text((150, 768), "product patterns, never on a handful", font=F(28, "s"), fill=C["amber"])
    if rv >= 2:
        mx = CF_EX[0][1]
        d.text((940, 290), "NEXT PRODUCT  (served engine, real numbers)", font=F(26, "mb"), fill=C["lav"])
        for k, (p, v) in enumerate(CF_EX):
            y = 360 + k * 95
            d.text((940, y), p, font=F(32, "b"), fill=C["white"])
            w = 700 * v / mx
            d.rounded_rectangle([940, y + 45, 940 + 700, y + 75], radius=15, fill=C["card2"])
            d.rounded_rectangle([940, y + 45, 940 + max(w, 30), y + 75], radius=15, fill=C["bright"])
            d.text((1660, y + 40), pc(v), font=F(32, "mb"), fill=C["white"])
    if rv >= 4:
        d.rounded_rectangle([940, 760, W - 110, 860], radius=20, fill=C["card"], outline=C["mint"], width=3)
        d.text((975, 790), f"ranked by surprise (lift): {pc(N['cf_old'])}   →   by probability: {pc(N['cf'])}",
               font=F(30, "b"), fill=C["mint"])


def r_fusion(img, d, data, rv):
    box(d, (110, 300, 560, 470), "35 profile + 48 history", "holdings, who they are, how they got here", "lav", rv >= 2)
    box(d, (110, 560, 560, 730), "44 peer scores", "from the collaborative engine", "violet", rv >= 2)
    box(d, (720, 380, 1250, 650), "Content engine", "gradient boosting\nreads all 127 inputs", "bright", rv >= 1, 44)
    box(d, (1410, 430, 1810, 600), "22 scores", "one per product", "mint", rv >= 1)
    arrow(d, 565, 385, 715, 470, rv >= 2)
    arrow(d, 565, 645, 715, 560, rv >= 2)
    arrow(d, 1255, 515, 1405, 515, rv >= 1)
    if rv >= 3:
        d.text((985, 690), "learns per customer how much to trust the peers", font=F(30, "s"), fill=C["amber"], anchor="ma")
    if rv >= 4:
        d.rounded_rectangle([110, 770, 900, 870], radius=20, fill=C["card"], outline=C["rose"], width=3)
        d.text((140, 795), f"first try — 50/50 average: {pc(N['blend_old'])}", font=F(30, "b"), fill=C["rose"])
        d.text((140, 832), f"worse than content alone ({pc(N['content'])})", font=F(26), fill=C["dim"])
        d.rounded_rectangle([960, 770, W - 110, 870], radius=20, fill=C["card"], outline=C["mint"], width=3)
        d.text((990, 795), f"fusion: {pc(N['nh'])}  →  + history: {pc(N['fused'])}", font=F(30, "b"), fill=C["mint"])
        d.text((990, 832), "the best of every engine tried", font=F(26), fill=C["dim"])


def r_macro(img, d, data, rv):
    series = ["XLE", "XLK", "EEM", "AAPL", "MSFT", "JPM", "BAC", "GS", "Brent"]
    on = rv >= 1
    for k, s in enumerate(series):
        x = 110 + k * 120
        d.rounded_rectangle([x, 280, x + 105, 380], radius=16, fill=a(C["card"], on), outline=a(C["sky"], on), width=2)
        d.text((x + 52, 330), s, font=F(26, "b"), fill=a(C["white"], on), anchor="mm")
    d.text((110, 405), "each series votes: momentum (6 months) + volatility", font=F(28), fill=a(C["dim"], on))
    regimes = [("EXPANSION", "mint"), ("RECOVERY", "sky"), ("SLOWDOWN", "amber"), ("STRESS", "rose")]
    for k, (r, col) in enumerate(regimes):
        x = 110 + k * 280
        d.rounded_rectangle([x, 470, x + 260, 550], radius=40, fill=a(C[col], on))
        d.text((x + 130, 510), r, font=F(28, "b"), fill=a(C["bg"], on), anchor="mm")
    if rv >= 2:
        d.rounded_rectangle([1250, 270, W - 110, 690], radius=26, fill=C["card"], outline=C["rose"], width=3)
        d.text((1290, 300), "IN STRESS", font=F(28, "mb"), fill=C["rose"])
        tilts = [("current account", "× 1.30", "mint"), ("short-term deposits", "× 1.25", "mint"),
                 ("investment funds", "× 0.70", "rose"), ("mortgage", "× 0.60", "rose")]
        for k, (p, t, col) in enumerate(tilts):
            d.text((1290, 360 + k * 75), p, font=F(32, "s"), fill=C["white"])
            d.text((W - 150, 360 + k * 75), t, font=F(34, "mb"), fill=C[col], anchor="ra")
    if rv >= 3:
        d.rounded_rectangle([110, 640, 1180, 860], radius=26, fill=C["card"], outline=C["amber"], width=3)
        d.text((150, 670), f"changes the #1 product for {pc(N['tilt'])} of customers", font=F(36, "b"), fill=C["amber"])
        d.text((150, 730), "top three unchanged — it adds business sense,", font=F(30), fill=C["dim"])
        d.text((150, 772), "not accuracy, and we report it that way", font=F(30), fill=C["dim"])


def r_rules(img, d, data, rv):
    rules = [("mortgage", "age 25–65 · income ≥ 30,000 · 2 years with the bank"),
             ("investment funds", "income ≥ 35,000 (suitability)"),
             ("pensions", "requires a payroll account")]
    for k, (p, rule) in enumerate(rules):
        on = rv >= 1
        y = 270 + k * 120
        d.rounded_rectangle([110, y, 1180, y + 100], radius=20, fill=a(C["card"], on), outline=a(C["line"], on), width=2)
        d.text((150, y + 18), p, font=F(34, "b"), fill=a(C["white"], on))
        d.text((150, y + 60), rule, font=F(26), fill=a(C["dim"], on))
    chips = [("removed, never down-weighted", "rose"), ("missing data never passes", "amber"),
             ("13 rules a compliance officer can read", "sky")]
    for k, (t, col) in enumerate(chips):
        on = rv >= 2
        y = 650 + k * 70
        d.rounded_rectangle([110, y, 900, y + 56], radius=28, fill=a(C["card"], on), outline=a(C[col], on), width=3)
        d.text((140, y + 28), t, font=F(28, "s"), fill=a(C[col], on), anchor="lm")
    if rv >= 3:
        d.rounded_rectangle([1250, 270, W - 110, 860], radius=26, fill=C["card"], outline=C["rose"], width=3)
        d.text((1300, 310), "THE COST", font=F(28, "mb"), fill=C["rose"])
        d.text((1300, 380), pc(N["removed"]), font=F(120, "b"), fill=C["rose"])
        for k, ln in enumerate(["of the time the rules remove", "the product the customer", "actually took — measured,", "reported, not hidden"]):
            d.text((1300, 560 + k * 48), ln, font=F(30), fill=C["dim"])


def r_image(img, d, data, rv):
    shot = Image.open(os.path.join(ASSETS, data["img"])).convert("RGB")
    bw, bh = 1680, 680
    s = min(bw / shot.width, bh / shot.height)
    shot = shot.resize((int(shot.width * s), int(shot.height * s)), Image.LANCZOS)
    x = (W - shot.width) // 2
    y = 245
    mask = Image.new("L", shot.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, shot.width - 1, shot.height - 1], radius=22, fill=255)
    d.rounded_rectangle([x - 6, y - 6, x + shot.width + 6, y + shot.height + 6], radius=26, fill=C["line"])
    img.paste(shot, (x, y), mask)


def r_bars(img, d, data, rv):
    rows = [("random 3 picks", 3 / 22, "mute", 1), ("popularity list", N["pop"], "mute", 2),
            ("collaborative (first, lift)", N["cf_old"], "rose", 3), ("collaborative (improved)", N["cf"], "violet", 3),
            ("content engine alone", N["content"], "violet", 3), ("50/50 blend (first hybrid)", N["blend_old"], "rose", 3),
            ("fused, no history (v2)", N["nh"], "violet", 4),
            ("FUSED + HISTORY", N["fused"], "mint", 5), ("full system, after rules", N["pipe"], "amber", 5)]
    y = 250
    for name, v, col, need in rows:
        on = rv >= need
        bold = "FUSED" in name or "full" in name
        d.text((110, y + 8), name, font=F(30, "b" if bold else "s"), fill=a(C["white"], on))
        d.rounded_rectangle([640, y + 4, 1640, y + 52], radius=24, fill=a(C["card"], on))
        d.rounded_rectangle([640, y + 4, 640 + max(1000 * v, 48), y + 52], radius=24, fill=a(C[col], on))
        d.text((1665, y + 6), pc(v), font=F(34, "mb"), fill=a(C["white"], on))
        y += 70
    if rv >= 6:
        d.rounded_rectangle([1290, 140, W - 110, 205], radius=18, fill=C["card"], outline=C["lav"], width=2)
        d.text((1320, 153), f"macro F1 {N['f1']:.3f}  ·  ~8× chance", font=F(30, "b"), fill=C["lav"])


def r_compare(img, d, data, rv):
    pairs = [("right product in top 3", N["nh"], N["fused"], 1, pc),
             ("first suggestion right (top 1)", N["nh_top1"], N["top1"], 2, pc),
             ("macro F1  (rare products count equally)", N["nh_f1"], N["f1"], 3, lambda v: f"{v:.3f}")]
    y = 280
    for name, a_, b_, need, fmt in pairs:
        on = rv >= need
        d.rounded_rectangle([110, y, W - 110, y + 175], radius=24, fill=a(C["card"], on), outline=a(C["line"], on), width=2)
        d.text((150, y + 25), name, font=F(34, "b"), fill=a(C["white"], on))
        d.text((150, y + 90), "v2, no history", font=F(26), fill=a(C["dim"], on))
        d.text((150, y + 122), fmt(a_), font=F(40, "mb"), fill=a(C["lav"], on))
        arrow(d, 560, y + 130, 700, y + 130, on, "mint")
        d.text((760, y + 90), "with 3 months of history", font=F(26), fill=a(C["dim"], on))
        d.text((760, y + 122), fmt(b_), font=F(40, "mb"), fill=a(C["mint"], on))
        gain = (b_ - a_) * 100 if fmt is pc else (b_ / a_ - 1) * 100
        lab = f"+{gain:.1f} pts" if fmt is pc else f"+{gain:.0f}%"
        d.text((W - 150, y + 100), lab, font=F(56, "b"), fill=a(C["amber"], on), anchor="ra")
        y += 200


def r_summary(img, d, data, rv):
    r_bullets(img, d, data, rv)


RENDER = dict(title=r_title, bullets=r_bullets, cards=r_cards, events=r_events, attributes=r_attributes,
              timeline=r_timeline, flow=r_flow, cf=r_cf, fusion=r_fusion, macro=r_macro, rules=r_rules,
              image=r_image, bars=r_bars, compare=r_compare, summary=r_summary)


# ================================================================== speech
def speak(items):
    """items: [(text, wav_path)] -> Windows speech, offline, one run."""
    tsv = os.path.join(BUILD, "lines.tsv")
    body = "".join(path + "\t" + text.replace("\t", " ") + "\n" for text, path in items)
    if (os.path.exists(tsv) and open(tsv, encoding="utf-8").read() == body
            and all(os.path.exists(p) for _, p in items)):
        print("  narration unchanged -- reusing the recorded voice")
        return
    with open(tsv, "w", encoding="utf-8") as f:
        f.write(body)
    ps = os.path.join(BUILD, "speak.ps1")
    with open(ps, "w", encoding="utf-8") as f:
        f.write(f'''Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("{VOICE}")
$s.Rate = {RATE}
foreach ($line in [IO.File]::ReadAllLines("{tsv}", [Text.Encoding]::UTF8)) {{
  $p = $line.Split("`t", 2)
  $s.SetOutputToWaveFile($p[0])
  $s.Speak($p[1])
}}
$s.SetOutputToNull()
''')
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps], check=True)


def wav_info(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate(), w.getparams()


def srt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


# ==================================================================== main
def main():
    global BASE
    os.makedirs(BUILD, exist_ok=True)
    BASE = gradient()
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    steps = []
    for si, sc in enumerate(SCENES):
        for k, (text, rv) in enumerate(sc["steps"]):
            steps.append((si, k, text, rv))
    print(f"{len(SCENES)} scenes, {len(steps)} narrated sentences")

    print("rendering frames ...")
    for si, k, text, rv in steps:
        frame(si, SCENES[si], rv, text).save(os.path.join(BUILD, f"f_{si:02d}_{k:02d}.png"))

    print("recording voice-over ...")
    speak([(text, os.path.join(BUILD, f"a_{si:02d}_{k:02d}.wav")) for si, k, text, rv in steps])

    print("assembling ...")
    params, chunks, concat, srt, t = None, [], [], [], 0.0
    for n, (si, k, text, rv) in enumerate(steps):
        wp = os.path.join(BUILD, f"a_{si:02d}_{k:02d}.wav")
        dur, p = wav_info(wp)
        params = params or p
        last_of_scene = n + 1 == len(steps) or steps[n + 1][0] != si
        pad = GAP + (SCENE_GAP if last_of_scene else 0)
        with wave.open(wp) as w:
            chunks.append(w.readframes(w.getnframes()))
        chunks.append(b"\x00" * int(pad * params.framerate) * params.sampwidth * params.nchannels)
        concat.append(f"file '{os.path.join(BUILD, f'f_{si:02d}_{k:02d}.png')}'\nduration {dur + pad:.3f}")
        srt.append(f"{n + 1}\n{srt_time(t)} --> {srt_time(t + dur)}\n{text}\n")
        t += dur + pad
    concat.append(f"file '{os.path.join(BUILD, f'f_{steps[-1][0]:02d}_{steps[-1][1]:02d}.png')}'")
    audio = os.path.join(BUILD, "voice.wav")
    with wave.open(audio, "wb") as w:
        w.setparams(params)
        w.writeframes(b"".join(chunks))
    lst = os.path.join(BUILD, "frames.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write("\n".join(c.replace("\\", "/") for c in concat) + "\n")
    with open(os.path.splitext(OUT)[0] + ".srt", "w", encoding="utf-8") as f:
        f.write("\n".join(srt))

    vf = f"fps=30,format=yuv420p,fade=t=in:st=0:d=0.8,fade=t=out:st={t - 1.2:.2f}:d=1.2"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", audio,
                    "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "stillimage",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", OUT], check=True)
    print(f"\nDONE  {OUT}  ({t / 60:.1f} min, {os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
