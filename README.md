# Polyrhythmia
`A modular, generative polyrhythmic arpeggiator`

**Polyrhythmia** is a Python-based generative arpeggiator and audio synthesizer that renders a 64-bar, multi-layered harmonic progression with modal modulations and algorithmic textures. It blends music theory with synthesis and gives you a full ambient progression directly rendered to `.wav` or played live.

---

## 🧰 Features

- 🎼 64-bar evolving **chord progression** across four musical "movements"
- 🎹 Multiple **arpeggiation styles**: walk, outward, updown, melody, etc.
- 🧠 **Music theory-aware** scale and chord selection (via `mingus`)
- 🔊 Layered **polyphonic synthesis** with:
  - Sine oscillators
  - Organ-style harmonic stacks
  - Square wave detuned textures
- 🎛️ Mastering chain with:
  - DC offset removal
  - Low-pass filtering
  - Normalization and headroom control
- 📦 Output to `.wav` and optional **live playback**
- ⌨️ Console display of harmonic sections and chord flow in real time

---

## 🚀 Requirements

Install dependencies via `pip`:

```bash

pip install -r requirements.txt
```

---

## ▶️ Usage

Simply run the script:

```bash
python polyrhythmia.py
```

- A `.wav` file named `song.wav` will be generated.
- The audio will be played back live if your system supports `simpleaudio`.
- The console will display the current section and chords in real time.

---

## 🧠 Musical Structure

The piece is divided into four parts:

1. **Invocation** – E minor modal incantation
2. **Spiral Ascent** – Bright G major ascent through modal cycles
3. **Fractured Horizons** – Chromatic and modal instability
4. **Homecoming Reverie** – Resolution and return to E minor

Each section contains 16 chords, with a tempo of **50 BPM** and each chord lasting **2 beats**.

---

## 🎛️ Synth Engine

You can choose between several oscillator styles:

- `synth_note` – Pure sine wave with exponential decay
- `organ_synth_note` – Additive harmonics and slight vibrato/detuning
- `square_synth_note` – Detuned square wave with optional envelope

Modify the function called inside `make_layer_sound()` to change the timbre.

---

## 🎹 Layer Configuration

Defined in `LAYER_CONFIGS`, each voice (Soprano, Alto, Tenor, Bass) has:

- `division`: how many notes per chord
- `octave`: base octave
- `pattern`: the arpeggiation pattern (`walk`, `outward`, `melody`, etc.)
- `name`: label for display purposes

---

## 🔊 Output & Mastering

Final audio is:

- Mixed across layers
- Filtered (5 kHz low-pass)
- DC-offset corrected
- Normalized to 90% peak

And then saved and optionally played.

---

## 📝 Customization

Feel free to:
- Replace the `CHORD_SEQUENCE` with your own harmonic ideas
- Add new arpeggiation patterns
- Introduce dynamics, rests, or velocity variation
- Try stereo panning or reverb for spatial depth

---

## 📁 Output

- `harmonic_journey.wav`: 44.1 kHz stereo file, ~150 seconds long

---

## 🧠 Credits & Acknowledgments

- Built with love for generative music, ambient textures, and algorithmic creativity.
- Uses the [`mingus`](https://github.com/bspaans/python-mingus) library for music theory utilities.

---

## 📜 License

MIT License. Use it, remix it, perform it, or bend it to your will.

---

🎧 _"Code is your baton. Conduct the air."_
