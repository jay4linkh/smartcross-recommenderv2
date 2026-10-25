#!/usr/bin/env bash
# Screenshots of the running console (python web/backend/app.py) for the video.
# Needs the temporary page web/frontend/_capture.html and Microsoft Edge.
set -e
E="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
OUT="$(cd "$(dirname "$0")" && pwd)/assets"
mkdir -p "$OUT"
shot() {   # name  query
  "$E" --headless=new --disable-gpu --hide-scrollbars --window-size=1600,1000 \
       --virtual-time-budget=15000 --force-prefers-reduced-motion \
       --screenshot="$(cygpath -w "$OUT")\\$1.png" \
       "http://127.0.0.1:8000/static/_capture.html?$2" >/dev/null 2>&1
  echo "  $1.png"
}
shot app_story_walk   "story=0.13"
shot app_story_chips  "story=0.52"
shot app_story_engine "story=0.79"
shot app_story_final  "story=0.985"
shot app_why          "draw=1&el=.card.why&dy=-20"
shot app_recs         "draw=1&el=%23verdict&dy=380"
shot app_engines      "draw=1&view=engines&node=Content%20engine&el=%23flow&dy=-110"
shot app_proof        "view=performance&el=%23evidence-box&dy=-10"
shot app_perf         "view=performance"
shot app_controls     "draw=1&el=%23regimes&dy=-160"
shot app_hero         "none=1"
