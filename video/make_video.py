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
ENGINE = "edge"                        # "edge": natural neural voice (online) | "sapi": Windows voice (offline)
EDGE_VOICE = "en-US-AndrewMultilingualNeural"   # also good: en-US-AvaMultilingualNeural, en-IN-PrabhatNeural
EDGE_RATE = "+4%"
VOICE = "Microsoft Zira Desktop"      # SAPI fallback: or "Microsoft David Desktop"
RATE = 1                               # SAPI: -10 (slow) .. 10 (fast)
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
        ("Imagine walking into your bank, and before you even ask, they already know what you'll need next. That's SmartCross.", 1),
        ("In the next few minutes: where the data comes from, how the model thinks, and how often it gets it right.", 2),
    ]),
    dict(kind="bullets", chapter="01 · THE PROBLEM", title="Offer the right product, not every product", data={"items": [
        ("Same list for everyone", "Most banks show every customer the same products, and most people ignore them."),
        ("Existing customers are the cheapest growth", "Selling a second product is far cheaper than finding a brand-new customer."),
        ("Our goal", "For each customer: the three products they're most likely to want next, only ones they qualify for, each explained."),
    ]}, steps=[
        ("Today, most banks show everyone the same list of products, and most people simply ignore it.", 1),
        ("But selling a second product to someone who already banks with you is much cheaper than finding a new customer.", 2),
        ("So our goal is simple. For every customer, pick the three products they're most likely to want next, skip anything they can't get, and explain why.", 3),
    ]),
    dict(kind="cards", chapter="02 · THE DATA", title="Three sources of data", data={"cards": [
        ("Santander bank", "2.2 GB", ["13.6 million rows", "24 products, month by month", "Jan 2015 – May 2016", "the main dataset"], "violet"),
        ("LendingClub", "1.6 GB", ["2.26 million US loans", "credit attributes", "Part A only:", "the first experiment"], "mute"),
        ("Market prices", "1 MB", ["9 price series", "sector funds, bank stocks", "monthly", "reads the economy"], "sky"),
    ]}, steps=[
        ("We used three kinds of data.", 0),
        ("The main one comes from Santander bank: 13.6 million rows, showing which of 24 products each customer had, every single month, for a year and a half.", 1),
        ("We also tried LendingClub loan data first. More on why it didn't work in a moment.", 2),
        ("And nine stock market price series, which tell us whether the economy is doing well or struggling.", 3),
    ]),
    dict(kind="bullets", chapter="02 · THE DATA", title="Lesson one: credit data isn't enough", data={"items": [
        ("The question", "Can credit data predict which loan a person wants?"),
        ("The answer: no", "Income and credit score show how much someone can borrow, not what they want. Barely better than guessing."),
        ("The lesson", "We need each customer's story over time. Santander has exactly that."),
    ]}, steps=[
        ("LendingClub only tells you about one loan per person.", 1),
        ("Income and credit score tell you how much someone can borrow, but not what they actually want. Our best model there was barely better than guessing.", 2),
        ("What we really needed was each customer's story over time. And that's exactly what Santander has.", 3),
    ]),
    dict(kind="events", chapter="03 · FROM RAW DATA TO LESSONS", title="Every 'no → yes' is a lesson", data={}, steps=[
        ("Here's the trick that turns raw data into something a model can learn from.", 0),
        ("Every customer has one row per month: a simple yes or no for every product.", 1),
        ("When a no turns into a yes, that customer just signed up for something new.", 2),
        ("Each of those moments becomes a lesson: what they had, who they are, and what they picked. We found 563,310 of them.", 3),
    ]),
    dict(kind="attributes", chapter="04 · WHAT THE MODEL LOOKS AT", title="35 clues + 3 months of history", data={}, steps=[
        ("So what does the model actually look at?", 0),
        ("First, which products the customer already has, like a current account, a payroll account, or a credit card.", 1),
        ("Then, who they are: their age, income, how long they've banked here, and how active they are.", 2),
        ("What it learns to predict: which of 22 products they took next.", 3),
        ("And new in this version: their recent history. What they had two months ago, and how long they've kept each product. That one idea made the biggest difference of all.", 4),
    ]),
    dict(kind="timeline", chapter="05 · A FAIR TEST", title="Study, practice test, final exam", data={}, steps=[
        ("To keep it honest, we split the data by time, just like a real exam.", 0),
        (f"The model studies everything from February 2015 to February 2016: {N['train']:,} lessons.", 1),
        ("February 2016 is the practice test. We only use it to choose settings.", 2),
        (f"And March to May 2016 is the final exam: {N['test']:,} customers the model has never seen.", 3),
    ]),
    dict(kind="flow", chapter="06 · HOW IT WORKS", title="Four steps, always in this order", data={}, steps=[
        ("The whole system works in four steps.", 0),
        ("First, it builds a profile of the customer, including their recent history.", 1),
        ("Second, two engines score every product.", 2),
        ("Third, it checks the economy.", 3),
        ("Fourth, it removes anything the customer isn't allowed to get.", 4),
        ("What's left is the top three, each with a reason a person can understand.", 5),
    ]),
    dict(kind="cf", chapter="07 · ENGINE 1 — THE CROWD", title="What did people like you do next?", data={}, steps=[
        ("Engine one asks a really simple question: what did people like you do next?", 1),
        (f"For example, people with a current account, a payroll account and direct debit most often went on to add {CF_EX[0][0]} or {CF_EX[1][0]}.", 2),
        ("If almost nobody shares your exact combination, it falls back on simpler patterns, so it never trusts just a handful of people.", 3),
        (f"Our first version chased rare, surprising products and was right only {N['cf_old']*100:.0f} percent of the time. Once we fixed that, it's right {N['cf']*100:.0f} percent of the time.", 4),
    ]),
    dict(kind="fusion", chapter="08 · ENGINE 2 — THE BRAIN", title="Learning when to trust the crowd", data={}, steps=[
        ("Engine two is the clever one: a machine-learning model made of thousands of small decision trees.", 1),
        ("It reads everything: the profile, the recent history, and engine one's opinion. That's 127 clues in total.", 2),
        ("Instead of just averaging the two engines, it learns when to trust the crowd, and when to trust the person in front of it.", 3),
        ("Our first attempt simply averaged them, and it actually made things worse. Lesson learned.", 4),
    ]),
    dict(kind="macro", chapter="09 · STEP 3 — THE ECONOMY", title="The economy gets a vote", data={}, steps=[
        ("Step three: the economy gets a vote.", 0),
        ("Nine market signals vote on whether we're in a boom, a recovery, a slowdown, or a period of stress.", 1),
        ("In tough times, safe products move up the list, and risky ones move down.", 2),
        (f"It changes the top pick for only {N['tilt']*100:.1f} percent of customers. It's there for good sense, not for accuracy, and we say so.", 3),
    ]),
    dict(kind="rules", chapter="10 · STEP 4 — THE RULES", title="If you don't qualify, it's gone", data={}, steps=[
        ("Step four: the rules.", 0),
        ("Thirteen simple rules. For example, a mortgage needs an age between 25 and 65, an income of at least 30,000, and two years with the bank.", 1),
        ("If you don't qualify, the product is removed, not just pushed down. And missing information never counts as a yes.", 2),
        (f"That has a cost: about {N['removed']*100:.0f} percent of the time, the rules remove the very thing the customer bought. We report that openly.", 3),
    ]),
    dict(kind="image", chapter="11 · SEE IT WORKING", title="Open the app", data={"img": "app_hero.png"}, steps=[
        ("Now let's see it working. Run the app, open it in your browser, and this is the first screen.", 1),
        ("At the top: how often it's right, how many real customers you can test it on, and how many products it ranks.", 1),
    ]),
    dict(kind="image", chapter="11 · SEE IT WORKING", title="Scroll: a customer walks in", data={"img": "app_story_walk.png"}, steps=[
        ("Scroll down, and a real customer from the final exam walks into the bank. As you keep scrolling, you follow them through every step.", 1),
    ]),
    dict(kind="image", chapter="11 · SEE IT WORKING", title="Watch the engines decide", data={"img": "app_story_engine.png"}, steps=[
        ("You can watch each step change the scores: what the crowd said, what the model concluded, and the economy's little nudge.", 1),
        ("And at the end, the app checks its answer against what this customer really did next.", 1),
    ]),
    dict(kind="image", chapter="11 · SEE IT WORKING", title="Draw a real customer", data={"img": "app_recs.png"}, steps=[
        ("Further down, press Draw a real customer. The app picks someone it has never seen, and shows its three recommendations, each with a reason and a score.", 1),
        ("At the top it tells you straight away whether it got it right, by comparing with what that customer actually did.", 1),
    ]),
    dict(kind="image", chapter="11 · SEE IT WORKING", title="Why this answer", data={"img": "app_why.png"}, steps=[
        ("Then read Why this answer: what we recommend and why, what was removed and by which rule, and what else was considered, all in plain English.", 1),
    ]),
    dict(kind="image", chapter="11 · SEE IT WORKING", title="Change anything, watch it update", data={"img": "app_controls.png"}, steps=[
        ("On the left, you can change anything about the customer: tick the products they hold, or edit their age, income or segment.", 1),
        ("Further down are controls for the economy, and a slider between the model and the crowd. Change any of them, and the recommendations update live, with a note telling you what moved.", 1),
    ]),
    dict(kind="image", chapter="11 · SEE IT WORKING", title="Inside the engines", data={"img": "app_engines.png"}, steps=[
        ("Finally, Inside the Engines shows the whole system at a glance, with a real, measured number on every box.", 1),
    ]),
    dict(kind="bars", chapter="13 · HOW GOOD IS IT?", title="Right product in our top three?", data={}, steps=[
        ("So, how good is it? Was the product they actually bought in our top three?", 1),
        (f"Picking three products at random would be right about 14 percent of the time. Just recommending the most popular products gets {N['pop']*100:.0f} percent.", 2),
        (f"The crowd engine alone gets {N['cf']*100:.0f} percent, and the profile model alone {N['content']*100:.0f}.", 3),
        (f"Combining the two, without history, also gets {N['nh']*100:.0f}.", 4),
        (f"Add the customer's recent history, and it climbs past {N['fused']*100:.0f} percent. Even after the strict rules, the full system is right {N['pipe']*100:.0f} percent of the time.", 5),
        ("And on rare products, which are the hardest, it does almost eight times better than chance.", 6),
    ]),
    dict(kind="compare", chapter="13 · HOW GOOD IS IT?", title="What three months of history added", data={}, steps=[
        ("Here's what that one idea, the customer's history, really added.", 1),
        (f"The very first suggestion is now right {N['top1']*100:.1f} percent of the time, up from {N['nh_top1']*100:.1f}.", 2),
        ("And on rare products, the score improved by 28 percent.", 3),
    ]),
    dict(kind="image", chapter="14 · IS IT REAL?", title="Real, or just luck?", data={"img": "app_proof.png"}, steps=[
        ("But is this real, or just luck?", 1),
        ("We re-checked the results a thousand different ways, and every gain from history held up every time.", 1),
        ("We also tried four other ideas, including training on a graphics card. None of them beat this model, and we report every one of them.", 1),
        ("It's steady from month to month, and when it says it's 60 percent sure, it's right about 60 percent of the time.", 1),
    ]),
    dict(kind="summary", chapter="IN SHORT", title="SmartCross in one slide", data={"items": [
        ("Two engines + recent history", "The crowd and the model, working together, learned from 563,310 real decisions"),
        ("Economy check + strict rules", "Every suggestion checked, and explained in plain English"),
        (f"{pc(N['pipe'])} right in the top three", "vs about 14% by chance and 70.5% for the popular list"),
    ]}, steps=[
        ("So, in short.", 0),
        ("Two engines working together, plus each customer's recent history, learned from 563 thousand real decisions.", 1),
        ("A common-sense economy check and strict rules on top, with every suggestion explained.", 2),
        (f"And {N['pipe']*100:.0f} percent of the time, what the customer actually bought next was right there in our top three. That's SmartCross. Thanks for watching.", 3),
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
    items = [("Profile", "35 numbers\n+ 48 history", "lav"), ("Two engines", "collaborative +\ncontent, fused", "violet"),
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
    box(d, (110, 300, 560, 470), "35 profile + 48 history", "who they are + their history", "lav", rv >= 2)
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
    """items: [(text, wav_path)] -> one wav per sentence, natural voice or Windows."""
    if ENGINE == "edge":
        return speak_edge(items)
    return speak_sapi(items)


def speak_edge(items):
    """Microsoft neural voice via edge-tts (online), converted to one wav format."""
    import asyncio
    import edge_tts
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tsv = os.path.join(BUILD, "lines.tsv")
    body = f"{EDGE_VOICE}|{EDGE_RATE}\n" + "".join(path + "\t" + text + "\n" for text, path in items)
    if (os.path.exists(tsv) and open(tsv, encoding="utf-8").read() == body
            and all(os.path.exists(p) for _, p in items)):
        print("  narration unchanged -- reusing the recorded voice")
        return

    async def run():
        for text, path in items:
            mp3 = path[:-4] + ".mp3"
            await edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE).save(mp3)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", mp3, "-ar", "24000", "-ac", "1",
                            "-sample_fmt", "s16", path], check=True)
    asyncio.run(run())
    with open(tsv, "w", encoding="utf-8") as f:
        f.write(body)


def speak_sapi(items):
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
