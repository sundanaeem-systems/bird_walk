# Bird Walk

An offline bird identifier for people who want to listen to the birds around them instead of staring at a screen.

Press Enter, put the laptop down, and listen for 15 seconds. Bird Walk tells you which bird it heard, how sure it is, and gives a short field note. Everything runs locally. No internet, no account, no API key.

Built for the Hacktoberfest Open-Source AI Challenge, Week 1: Touch Grass.

## How it works

1. **Record** 15 seconds from the microphone (`sounddevice`).
2. **Where am I?** BirdNET's geo model (v2.4) takes latitude, longitude and the week of the year, and returns how likely each species is in that place right now. Only species above `GEO_MIN` are kept.
3. **What do I hear?** BirdNET's acoustic model (v2.4, running on LiteRT, no TensorFlow needed) scores the recording. For each candidate, audio confidence is multiplied by local likelihood, so a common local bird beats a rare bird that only sounds similar.
4. **Not sure is an answer.** Below `MIN_CONF` the tool says it is not sure instead of guessing.
5. **Field note.** Gemma 3 1B (open-weight, GGUF, via `llama-cpp-python`) writes two sentences about the bird. It is only allowed to rephrase a small table of local facts in `birdwalk.py` (`FACTS`). If a bird is not in the table, no note is printed, so a small model cannot invent things.

## Requirements

- Windows, macOS or Linux with a microphone
- Python 3.10 to 3.12 (developed on 3.12, Windows)
- About 1 GB of disk space for models

## Install

```bash
git clone https://github.com/YOUR_USERNAME/bird-walk.git
cd bird-walk

python -m venv venv
# Windows (CMD):        venv\Scripts\activate
# Windows (PowerShell): venv\Scripts\Activate.ps1
# macOS / Linux:        source venv/bin/activate

pip install -r requirements.txt
```

### Download the Gemma model

Download `gemma-3-1b-it-Q4_K_M.gguf` (about 800 MB) from
https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF
and put it in the `models/` folder:

```
bird-walk/
  models/
    gemma-3-1b-it-Q4_K_M.gguf
```

### BirdNET models

BirdNET downloads its acoustic and geo models automatically the first time you run the tool, so the first run needs internet. After that, everything works offline.

## Run

```bash
python birdwalk.py
```

Records 15 seconds from the microphone and prints the result.

To analyse an existing file instead:

```bash
python birdwalk.py some_recording.wav
```

Example output:

```
Attock, week 38: 469 species possible
Top candidates:
  House Sparrow: audio 0.87, local 0.xx

Bird: House Sparrow (Passer domesticus)  confidence: 87%

Field note:
A small bird with a grey crown and black bib is a common sight near houses. They are cheerful and often found in flocks.
```

## Settings

At the top of `birdwalk.py`:

| Setting | Meaning |
|---|---|
| `LAT, LON` | Your location. Default is Attock, Pakistan. Change it to yours. |
| `GEO_MIN` | Minimum local likelihood for a species to be considered (default 0.3). |
| `MIN_CONF` | Minimum audio confidence, below this it says "not sure" (default 0.4). |
| `SECONDS` | Recording length. |
| `FACTS` | Local facts the field note is allowed to use. Add birds from your area. |
| `SPEAK` | Optional voice output with Piper. Experimental and not tested yet. |

## Testing so far

- Played House Sparrow calls from a phone speaker next to the laptop microphone: identified correctly at roughly 80 to 95 percent across several runs.
- A recording with only human voices and background noise: "Not sure", as intended.
- Ran with Wi-Fi switched off: the full pipeline worked with no network access.
- Early version reported a Song Thrush at 77 percent, which is unlikely in this region in autumn. This is why the local filter and the confidence threshold were tightened.
- Not yet tested on a real outdoor walk.

## Limitations

- BirdNET can be confidently wrong, especially in noisy places.
- The facts table covers only about a dozen common birds in northern Pakistan.
- Gemma 1B does not know birds. That is why it only rephrases supplied facts.
- The week number is an approximation (4 weeks per month).
- Voice output is not finished.

## Why open models

- Works with no signal after the one-time download.
- Your exact location and microphone audio never leave your machine.
- No keys, fees or rate limits.
- You can change the system: filter the species list, change the facts, tune the thresholds, or swap the language model.

## Models and licences

- BirdNET (Cornell Lab of Ornithology and Chemnitz University of Technology): https://github.com/birdnet-team
- Gemma 3 (Google): open-weight, released under the Gemma terms of use. Please read them before redistributing.
- `llama-cpp-python`: MIT.

Check the licence of each model before any commercial use. The BirdNET models in particular may be restricted to non-commercial use.
