import numpy as np
import simpleaudio as sa
import time
import random
from mingus.core import chords, notes, scales
import scipy.io.wavfile
from scipy.signal import square, butter, sosfilt

# ---------- SETTINGS ----------
SAMPLE_RATE = 44100
BPM = 120
SECONDS_PER_BEAT = 60 / BPM
CHORD_DURATION_BEATS = 4  # Each chord lasts 2 beats
WINDOW_BEATS = 16
VOLUME = 0.8

# ---------- CHORD SEQUENCE ----------
# A 64-bar rich harmonic journey with modulations

SECTION_KEYS = {
    "I. Invocation": "E minor",
    "II. Ascent": "G major",
    "III. Disruption": "F major",
    "IV. Resolution": "E minor"
}

CHORD_SEQUENCE = [
    # I. Invocation - E minor modal intro
    "Em", "Em", "F#m7b5", "B7",
    "Em", "G", "Am7b5", "B7",
    "Cmaj7", "B7", "Em", "Em",
    "F#m7b5", "B7", "Em", "Em",

    # II. Ascent - G major elegance
    "Gmaj7", "Em7", "Am7", "D7",
    "Bm7", "E7", "Am7", "D7",
    "Gmaj7", "F#7", "Bm7", "Bm7",
    "Cmaj7", "G/B", "Am7", "D7",

    # III. Disruption - modal drift
    "Dm7", "G7", "Cmaj7", "Am7",
    "Bbmaj7", "C7", "Fmaj7", "Fmaj7",
    "Gm7", "C7", "Fmaj7", "A7",
    "Dm7", "G7", "Cmaj7", "D7",

    # IV. Resolution - Em coda
    "Am7", "D7", "Gmaj7", "Gmaj7",
    "F#m7b5", "B7", "Em", "Em9",
    "Em", "Cmaj7", "Am7", "B7",
    "Em", "F#m7b5", "B7", "Em"
]

TOTAL_BEATS = len(CHORD_SEQUENCE) * CHORD_DURATION_BEATS

# ---------- LAYER CONFIGS ----------
LAYER_CONFIGS = [
    dict(division=11, octave=5, pattern="walk", name="Soprano"),
    dict(division=10, octave=4, pattern="outward", name="Alto"),
    dict(division=7, octave=3, pattern="updown", name="Tenor"),
    dict(division=4, octave=2, pattern="walk", name="Bass"),
]

# ---------- AUDIO UTILITIES ----------
def note_to_freq(note, octave):
    m = notes.note_to_int(note)
    midi_number = m + (octave + 1) * 12
    freq = 440.0 * 2 ** ((midi_number - 69) / 12)
    return freq, midi_number

def synth_note(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2 * np.pi * freq * t)
    env = np.exp(-4 * t)
    fl = int(min(0.005, duration / 2) * SAMPLE_RATE)
    env[:fl] *= np.linspace(0, 1, fl)
    env[-fl:] *= np.linspace(1, 0, fl)
    wave *= env
    return (wave * volume).astype(np.float32)


def organ_synth_note(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Vibrato modulation
    vibrato = 0.002 * np.sin(2 * np.pi * 1 * t)  # 5 Hz vibrato

    # Dual detuned oscillators (±0.5%)
    wave1 = (
        1.0 * np.sin(2 * np.pi * freq * (t + vibrato)) +
        0.5 * np.sin(2 * np.pi * freq * 2 * (t + vibrato)) +
        0.3 * np.sin(2 * np.pi * freq * 3 * (t + vibrato)) +
        0.2 * np.sin(2 * np.pi * freq * 4 * (t + vibrato)) +
        0.1 * np.sin(2 * np.pi * freq * 5 * (t + vibrato))
    )

    wave2 = (
        1.0 * np.sin(2 * np.pi * freq * 1.005 * (t + vibrato)) +
        0.5 * np.sin(2 * np.pi * freq * 2 * 1.005 * (t + vibrato)) +
        0.3 * np.sin(2 * np.pi * freq * 3 * 1.005 * (t + vibrato))
    )

    wave = 0.5 * (wave1 + wave2)

    # Gentle release tail
    env = np.ones_like(t)
    release_time = 0.03
    release_samples = int(SAMPLE_RATE * release_time)
    if release_samples < len(env):
        env[-release_samples:] *= np.linspace(1, 0, release_samples)

    return (wave * volume).astype(np.float32)


def square_synth_note(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    # Vibrato
    vibrato = 0.002 * np.sin(2 * np.pi * 1 * t)

    # Square waves with slight detuning
    wave1 = square(2 * np.pi * freq * (t + vibrato))
    wave2 = square(2 * np.pi * freq * 1.005 * (t + vibrato))
    wave = 0.5 * (wave1 + wave2)

    # Apply envelope
    env = np.ones_like(t)
    release_time = 0.1
    release_samples = int(SAMPLE_RATE * release_time)
    if release_samples < len(env):
        env[-release_samples:] *= np.linspace(1, 0, release_samples)

    return (wave * env * volume).astype(np.float32)


def lowpass_filter(audio, cutoff_hz, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    norm_cutoff = cutoff_hz / nyquist
    sos = butter(order, norm_cutoff, btype='low', analog=False, output='sos')
    return sosfilt(sos, audio)


def remove_dc_offset(signal):
    return signal - np.mean(signal)


# ---------- ARPEGGIATOR ----------
def arpeggiate_notes(notes_list, division, pattern, chord_name=None, last_note=None):
    """Return a list of arpeggiated note names for a given pattern."""
    l = len(notes_list)
    if l == 0:
        return ["C"] * division  # fallback
    
    if pattern == "up":
        return (notes_list * ((division + l - 1) // l))[:division]
    
    elif pattern == "down":
        rev = list(reversed(notes_list))
        return (rev * ((division + l - 1) // l))[:division]
    
    elif pattern == "updown":
        updown = notes_list + list(reversed(notes_list[1:-1]))
        return (updown * ((division + len(updown) - 1) // len(updown)))[:division]
    
    elif pattern == "inward":
        # Outside-in: C–G–E for Cmaj (assuming [C,E,G])
        out_in = []
        for i in range((l+1)//2):
            if i < l - i:
                out_in += [notes_list[i], notes_list[-1 - i]]
            elif i == l - i:
                out_in += [notes_list[i]]
        return (out_in * ((division + len(out_in) - 1) // len(out_in)))[:division]
    
    elif pattern == "outward":
        # Inside-out: E–C–G for Cmaj (assuming [C,E,G])
        mid = l // 2
        out = [notes_list[mid]]
        for i in range(1, mid+1):
            if mid - i >= 0: out.append(notes_list[mid - i])
            if mid + i < l: out.append(notes_list[mid + i])
        return (out * ((division + len(out) - 1) // len(out)))[:division]
    
    elif pattern == "repeat":
        return [notes_list[0]] * division
    
    elif pattern == "random":
        return [random.choice(notes_list) for _ in range(division)]
    
    elif pattern == "walk":
        walk = [notes_list[random.randint(0, l-1)]]
        for _ in range(division - 1):
            idx = notes_list.index(walk[-1])
            step = random.choice([-1, 1])
            new_idx = (idx + step) % l
            walk.append(notes_list[new_idx])
        return walk
    elif pattern == "melody":
        scale_notes = chord_to_scale_notes(chord_name)
        melody = []
        rest_chance = 0.0  # 20% chance of rest

        for _ in range(division):
            if random.random() < rest_chance:
                melody.append(None)  # Represent a rest
                continue

            if not last_note or last_note not in scale_notes:
                note = random.choice(scale_notes)
            else:
                idx = scale_notes.index(last_note)
                step = random.choices([-2, -1, 1, 2], weights=[1, 3, 3, 1])[0]
                next_idx = (idx + step) % len(scale_notes)
                note = scale_notes[next_idx]
            melody.append(note)
            last_note = note
        return melody
    else:
        return [random.choice(notes_list) for _ in range(division)]  # fallback


def get_scale_for_chord_index(chord_idx):
    section = get_section_label(chord_idx)
    key_name = SECTION_KEYS.get(section, "C")
    if "minor" in key_name:
        root = key_name.split()[0]
        return scales.NaturalMinor(root).ascending()
    elif "major" in key_name:
        root = key_name.split()[0]
        return scales.Major(root).ascending()
    return scales.Major("C").ascending()


def chord_to_scale_notes(chord_name):
    """Return a scale (list of note names) that matches the chord context for melody."""
    try:
        chord_notes = chords.from_shorthand(chord_name)
        root = chord_notes[0]
    except:
        return ["C", "D", "E", "F", "G", "A", "B"]  # fallback

    root_midi = notes.note_to_int(root)
    scale_intervals = []

    # Define likely scale modes from chord type
    if "m7b5" in chord_name:
        scale_intervals = [0, 2, 3, 5, 6, 8, 10]  # Locrian
    elif "dim" in chord_name:
        scale_intervals = [0, 2, 3, 5, 6, 8, 9]   # Whole-Half dim
    elif "m" in chord_name and "maj" not in chord_name:
        scale_intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
    elif "7" in chord_name and "maj" not in chord_name:
        scale_intervals = [0, 2, 4, 5, 7, 9, 10]  # Mixolydian (dominant)
    elif "maj7" in chord_name or "maj" in chord_name:
        scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major (Ionian)
    else:
        scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Default major

    # Build scale notes
    scale_notes = []
    for i in scale_intervals:
        note_int = (root_midi + i) % 12
        scale_notes.append(notes.int_to_note(note_int))

    return scale_notes


def make_layer_sound(cfg, chord_seq, chord_dur, vol):
    seq = np.array([], dtype=np.float32)
    note_dur = chord_dur / cfg['division']
    last_note = None

    for chord in chord_seq:
        notes_list = chords.from_shorthand(chord)
        arp = arpeggiate_notes(notes_list, cfg['division'], cfg['pattern'], chord, last_note)
        for n in arp:
            if n is None:
                seq = np.concatenate((seq, np.zeros(int(SAMPLE_RATE * note_dur))))  # Insert silence
                continue
            f, _ = note_to_freq(n, cfg['octave'])
            seq = np.concatenate((seq, synth_note(f, note_dur, vol)))
            last_note = n
    return seq


def get_layer_hits(cfg, chord_seq):
    hits = []
    beat = 0.0
    for chord in chord_seq:
        notes_list = chords.from_shorthand(chord)
        arp = arpeggiate_notes(notes_list, cfg['division'], cfg['pattern'], chord)
        for i, n in enumerate(arp):
            pos = beat + i * (CHORD_DURATION_BEATS / cfg['division'])
            _, midi_number = note_to_freq(n, cfg['octave'])
            hits.append((pos, midi_number, n + str(cfg['octave'])))
        beat += CHORD_DURATION_BEATS
    return hits


def mix_tracks(tracks):
    ml = max(len(t) for t in tracks)
    padded = [np.pad(t, (0, ml - len(t))) for t in tracks]
    m = sum(padded)
    mx = np.max(np.abs(m))
    return (m / mx) * 0.9 if mx > 0 else m


def play_buffer(audio):
    buf = (audio * 32767).astype(np.int16)
    return sa.play_buffer(buf, 1, 2, SAMPLE_RATE)

# ---------- SAVE WAV ----------
def save_to_wav(filename, audio, sample_rate):
    audio_int16 = (audio * 32767).astype(np.int16)
    scipy.io.wavfile.write(filename, sample_rate, audio_int16)
    print(f"💾 Saved audio to '{filename}'")

# ---------- CONSOLE DISPLAY ----------
def get_section_label(chord_idx):
    if chord_idx < 16:
        return "I. Invocation"
    elif chord_idx < 32:
        return "II. Ascent"
    elif chord_idx < 48:
        return "III. Disruption"
    else:
        return "IV. Resolution"

last_section_printed = None

def print_console_frame(window_start):
    global last_section_printed
    chord_index = int(window_start // CHORD_DURATION_BEATS)

    # Print section header only if changed
    section = get_section_label(chord_index)
    if section != last_section_printed:
        print(f"\n🎼 {section}")
        print("-" * 32)
        last_section_printed = section

    # Print one line: 4 chords starting at chord_index
    line_chords = []
    for i in range(4):
        idx = chord_index + i
        if idx < len(CHORD_SEQUENCE):
            chord = CHORD_SEQUENCE[idx]
            line_chords.append(f"{chord:<7}")
        else:
            line_chords.append("   ---  ")

    print("  ".join(line_chords))

# ---------- MAIN ----------
if __name__ == "__main__":
    print("🎛️  Starting Harmonic Journey...")
    chord_dur = CHORD_DURATION_BEATS * SECONDS_PER_BEAT

    # Build audio layers
    layers = [
        make_layer_sound(cfg, CHORD_SEQUENCE, chord_dur, VOLUME)
        for cfg in LAYER_CONFIGS
    ]
    #audio = mix_tracks(layers)

    # Apply master filter
    audio = mix_tracks(layers)
    audio = lowpass_filter(audio, cutoff_hz=5000, sample_rate=SAMPLE_RATE)
    audio = remove_dc_offset(audio)
    audio /= np.max(np.abs(audio)) + 1e-9  # normalize and prevent div by zero
    audio *= 0.9  # add headroom

    # Save to wav file before playing
    save_to_wav("song.wav", audio, SAMPLE_RATE)

    # PLay audio
    player = play_buffer(audio)

    beat_sec = SECONDS_PER_BEAT
    total_duration = TOTAL_BEATS * beat_sec
    start = time.time()
    prev_win = -1

    while time.time() - start < total_duration:
        elapsed = time.time() - start
        bp = elapsed / beat_sec
        window_start = int(bp // WINDOW_BEATS) * WINDOW_BEATS
        if window_start != prev_win:
            print_console_frame(window_start)
            prev_win = window_start
        time.sleep(1)

    player.wait_done()
    print("✅ Journey complete.")
