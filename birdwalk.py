import sys
import datetime
import subprocess
import sounddevice as sd
import soundfile as sf
import birdnet
from llama_cpp import Llama

SECONDS = 15
SR = 48000
MIN_CONF = 0.4           # below this audio confidence, say "not sure"
GEO_MIN = 0.3            # minimum chance the bird occurs here this week
LAT, LON = 33.77, 72.36  # Attock
LOCATION = "Attock"      # only used for display
WAV = "recording.wav"
GEMMA = "models/gemma-3-1b-it-Q4_K_M.gguf"
SPEAK = False            # True = read the result aloud with Piper (optional)
VOICE = "en_US-lessac-medium"

# Local facts, keyed by a part of the bird's common name.
# The LLM may only use these facts, so it can't invent things.
FACTS = {
    "Common Myna": "Brown body with a black hooded head, bright yellow bill, eye patch and legs. Very loud: chattering, whistling and squeaky calls. Walks on the ground in pairs or groups in towns and fields.",
    "House Sparrow": "Small bird. Male has a grey crown and black bib, female is plain brown. Cheerful chirping calls. Lives close to houses, often in noisy flocks.",
    "Rose-ringed Parakeet": "Bright green with a long tail and red bill. Harsh, loud screeching calls, usually heard in flight in noisy flocks. Nests in tree holes.",
    "Red-vented Bulbul": "Dark head with a small crest and a red patch under the tail. Cheerful whistling calls. Common in gardens.",
    "Collared-Dove": "Pale grey-brown dove with a thin black half-collar on the neck. Soft repeated three-note coo. Often sits on wires and rooftops.",
    "Laughing Dove": "Small pinkish-brown dove with a spotted chest. Soft laughing coo. Common in towns.",
    "House Crow": "Black crow with a grey neck. Harsh 'kaa' call. Lives close to people and scavenges.",
    "Hoopoe": "Pinkish-brown with black-and-white striped wings and a large fan-shaped crest. Soft 'oop-oop-oop' call. Probes lawns for insects.",
    "Black Kite": "Dark brown bird of prey with a forked tail. Soars over towns and scavenges. Thin whistling call.",
    "Cattle Egret": "Small white heron that follows cattle in fields. Mostly silent.",
    "Indian Roller": "Brownish back, but flashes bright blue wings in flight. Often sits on wires. Harsh calls.",
    "Magpie-Robin": "Black and white with a long tail that is often cocked up. Sweet whistling song. Garden bird.",
    "Koel": "Male is glossy black with a red eye. Loud rising 'koo-ooo' call, heard mostly in spring and summer.",
}


def record(path, seconds=SECONDS):
    print(f"Recording {seconds} sec... listening now")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1)
    sd.wait()
    sf.write(path, audio, SR)
    print("Recording done.")


def current_week():
    t = datetime.date.today()
    return (t.month - 1) * 4 + min(4, (t.day - 1) // 8 + 1)   # BirdNET uses 48 weeks


def local_species():
    geo = birdnet.load("geo", "2.4", "tf", library="litert")
    week = current_week()
    df = geo.predict(LAT, LON, week=week).to_dataframe()
    df = df[df["confidence"] >= GEO_MIN]
    print(f"{LOCATION}, week {week}: {len(df)} species possible")
    return dict(zip(df["species_name"], df["confidence"]))


def detect(model, path, local):
    kw = dict(top_k=5, default_confidence_threshold=0.1)
    try:
        result = model.predict(path, custom_species_list=sorted(local), **kw)
    except Exception as e:
        print("(species list filter failed, filtering afterwards)", repr(e)[:120])
        result = model.predict(path, **kw)
    df = result.to_dataframe()
    df = df[df["species_name"].isin(local.keys())].copy()
    if df.empty:
        return None
    df["geo"] = df["species_name"].map(local)
    best = df.groupby("species_name").agg(conf=("confidence", "max"), geo=("geo", "first"))
    best["score"] = best["conf"] * best["geo"]
    best = best.sort_values("score", ascending=False)

    print("Top candidates:")
    for name, row in best.head(3).iterrows():
        print(f"  {name.split('_', 1)[1]}: audio {row['conf']:.2f}, local {row['geo']:.2f}")

    name = best.index[0]
    conf = float(best.iloc[0]["conf"])
    sci, common = name.split("_", 1)
    return common, sci, conf


def find_facts(common):
    for key, text in FACTS.items():
        if key.lower() in common.lower():
            return text
    return None


def field_note(llm, common, facts):
    prompt = (
        f"Bird: {common}.\nFacts: {facts}\n\n"
        "Write a friendly field note for a birdwatcher in exactly 2 short sentences. "
        "Reuse the exact words from the facts. Do not rephrase body parts or colours. "
        "Do not start with a greeting. Do not add anything else."
    )
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], max_tokens=120, temperature=0.0)
    text = out["choices"][0]["message"]["content"].strip()
    for greeting in ("Hello there!", "Hello!", "Hi there!", "Hi!", "Hey there!", "Okay,", "Okay!"):
        if text.startswith(greeting):
            text = text[len(greeting):].strip()
    return text[:1].upper() + text[1:]


def speak(text):
    # Optional: offline text-to-speech with Piper. Fails quietly if Piper isn't set up.
    try:
        subprocess.run([sys.executable, "-m", "piper", "-m", VOICE,
                        "-f", "speech.wav", "--", text], check=True)
        data, sr = sf.read("speech.wav")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print("(voice output failed:", repr(e)[:100], ")")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        path = WAV
        record(path)

    print("Loading models...")
    local = local_species()
    model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    llm = Llama(model_path=GEMMA, n_ctx=2048, verbose=False)

    found = detect(model, path, local)
    if found is None or found[2] < MIN_CONF:
        print("\nNot sure what that was. Move closer or reduce background noise and try again.")
        if found:
            print(f"(Closest guess: {found[0]}, confidence {found[2]:.2f}, below threshold)")
        return

    common, sci, conf = found
    print(f"\nBird: {common} ({sci})  confidence: {conf:.0%}")
    facts = find_facts(common)
    if facts:
        note = field_note(llm, common, facts)
        print("\nField note:")
        print(note)
        if SPEAK:
            speak(f"{common}. {note}")
    else:
        print("\n(No local notes for this bird yet.)")
        if SPEAK:
            speak(common)


if __name__ == "__main__":
    main()
