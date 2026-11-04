#!/bin/bash
# Generate clean TTS reference utterances (24 kHz mono s16) with macOS `say` and ffmpeg.
# usage: ./make_speech.sh [OUT_DIR]   (default: clean/ next to this script; 2 voices x 20 sentences)
set -euo pipefail
D="${1:-$(cd "$(dirname "$0")" && pwd)/clean}"
mkdir -p "$D"
SENTENCES=(
"Hello there, how can I help you today?"
"The weather this afternoon looks bright and sunny, with a light breeze coming in from the west."
"Sure, I can book that table for seven people at eight o'clock."
"Let me check that for you. It should only take a moment."
"Our store opens at nine in the morning and closes at six in the evening, Monday through Saturday."
"That's a great question."
"Photosynthesis is the process by which plants turn sunlight, water and carbon dioxide into sugar and oxygen."
"I'm sorry, I didn't quite catch that. Could you say it again?"
"Your package was shipped yesterday and should arrive by Thursday."
"Absolutely. Here are three quick tips for better sleep: keep a schedule, avoid screens, and keep the room cool."
"The quick brown fox jumps over the lazy dog."
"Mount Everest is the highest mountain above sea level, at about eight thousand eight hundred meters."
"Okay, I've set a timer for fifteen minutes."
"Would you like me to read the next chapter, or should we stop here for tonight?"
"Paris is the capital of France, and it is famous for the Eiffel Tower."
"Thanks for calling. Have a wonderful day!"
"To reset your password, open settings, choose account, and then tap reset password."
"Interesting! Tell me more about that."
"In nineteen sixty nine, astronauts landed on the moon for the very first time."
"Right, so the total comes to forty two dollars and fifty cents, including tax."
)
VOICES=("Samantha" "Daniel")
for v in "${VOICES[@]}"; do
  i=0
  for s in "${SENTENCES[@]}"; do
    i=$((i+1))
    n=$(printf "%s_%02d" "$v" "$i")
    say -v "$v" -o "$D/$n.aiff" "$s"
    ffmpeg -hide_banner -loglevel error -y -i "$D/$n.aiff" -ar 24000 -ac 1 -sample_fmt s16 "$D/$n.wav"
    rm "$D/$n.aiff"
  done
done
ls "$D" | wc -l
