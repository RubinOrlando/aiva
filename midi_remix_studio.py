#!/usr/bin/env python3
"""
MIDI Probability Remix Studio
=============================

A single-file web application that lets you:

  1. Upload a MIDI file (single track or multichannel).
  2. See **probability statistics** for every instrument it contains, both
     melodic (pitch classes, melodic intervals, note durations, velocity,
     pitch-transition Markov chain) and rhythmic / percussion (which drums
     are used, where in the bar they hit, inter-onset rhythm).
  3. Press a **Random Remix** button that re-composes every track into a new
     tune by sampling from that track's own probability templates (Markov
     chains + histograms), so the remix "sounds like" the original without
     copying it.
  4. Have the generated voices automatically reconciled with **counterpoint
     logic** so the parts line up harmonically: a key/scale is detected, all
     pitches are snapped to it, and each voice is nudged so it forms
     consonant intervals against the bass while avoiding parallel perfect
     fifths/octaves.

Only third-party dependency: `mido`  (`pip install mido`).
Everything else (web server, charts) uses the Python standard library.

Run it:

    python3 midi_remix_studio.py            # serves http://127.0.0.1:8765
    python3 midi_remix_studio.py --port 9000
    python3 midi_remix_studio.py --selftest # headless engine test, no browser

Then open the printed URL, drop in a .mid file, and hit "Random Remix".
"""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import io
import json
import math
import os
import random
import re
import sys
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Directory where every generated remix is auto-saved. Created on first use.
REMIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remixes")

try:
    import mido
except ImportError:  # pragma: no cover - friendly message
    sys.stderr.write(
        "This app needs the 'mido' package.\n"
        "Install it with:  pip install mido\n"
    )
    raise


# --------------------------------------------------------------------------- #
#  Music theory constants
# --------------------------------------------------------------------------- #

DRUM_CHANNEL = 9  # General MIDI percussion channel (0-indexed)
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Interval (mod 12) consonance classes used by the counterpoint engine.
# Against a bass, the perfect fourth (5) is treated as a dissonance, as in
# strict species counterpoint.
PERFECT_CONSONANCES = {0, 7}            # unison/octave, perfect fifth
IMPERFECT_CONSONANCES = {3, 4, 8, 9}    # thirds and sixths
CONSONANCES = PERFECT_CONSONANCES | IMPERFECT_CONSONANCES

# Scale templates as pitch-class offsets from the tonic.
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
NATURAL_MINOR = [0, 2, 3, 5, 7, 8, 10]

# Krumhansl-style key profiles for key detection (relative weights).
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# General MIDI program -> short instrument family name (coarse but useful).
GM_FAMILIES = [
    "Piano", "Chromatic Perc.", "Organ", "Guitar", "Bass", "Strings",
    "Ensemble", "Brass", "Reed", "Pipe", "Synth Lead", "Synth Pad",
    "Synth FX", "Ethnic", "Percussive", "Sound FX",
]

# General MIDI percussion key map (subset, for readable drum labels).
GM_DRUMS = {
    35: "Ac Bass Drum", 36: "Bass Drum 1", 37: "Side Stick", 38: "Ac Snare",
    39: "Hand Clap", 40: "El Snare", 41: "Low Floor Tom", 42: "Closed HiHat",
    43: "High Floor Tom", 44: "Pedal HiHat", 45: "Low Tom", 46: "Open HiHat",
    47: "Low-Mid Tom", 48: "Hi-Mid Tom", 49: "Crash 1", 50: "High Tom",
    51: "Ride 1", 52: "China", 53: "Ride Bell", 54: "Tambourine",
    55: "Splash", 56: "Cowbell", 57: "Crash 2", 59: "Ride 2",
    60: "Hi Bongo", 61: "Low Bongo", 62: "Mute Hi Conga", 63: "Open Hi Conga",
    64: "Low Conga", 69: "Cabasa", 70: "Maracas", 75: "Claves",
}


# --------------------------------------------------------------------------- #
#  Data model
# --------------------------------------------------------------------------- #

class Note:
    """A single sounding note in absolute ticks."""

    __slots__ = ("start", "dur", "pitch", "velocity")

    def __init__(self, start, dur, pitch, velocity):
        self.start = start
        self.dur = max(1, dur)
        self.pitch = pitch
        self.velocity = velocity

    @property
    def end(self):
        return self.start + self.dur


class Track:
    """One instrument's worth of notes, keyed by MIDI channel."""

    def __init__(self, channel, program, is_drum):
        self.channel = channel
        self.program = program
        self.is_drum = is_drum
        self.notes: list[Note] = []

    @property
    def name(self):
        if self.is_drum:
            return f"Channel {self.channel + 1} — Drum Kit"
        family = GM_FAMILIES[(self.program // 8) % 16] if self.program is not None else "Piano"
        return f"Channel {self.channel + 1} — {family}"

    @property
    def span(self):
        if not self.notes:
            return 0
        start = min(n.start for n in self.notes)
        end = max(n.end for n in self.notes)
        return end - start


# --------------------------------------------------------------------------- #
#  MIDI parsing
# --------------------------------------------------------------------------- #

def load_tracks(data: bytes):
    """Parse raw MIDI bytes into per-channel Track objects.

    Returns (tracks, ticks_per_beat, tempo).
    """
    mid = mido.MidiFile(file=io.BytesIO(data))
    tpb = mid.ticks_per_beat or 480
    tempo = 500000  # default 120 BPM until we see a set_tempo

    programs: dict[int, int] = defaultdict(int)
    open_notes: dict[tuple, tuple] = {}  # (channel, pitch) -> (start, velocity)
    tracks: dict[int, Track] = {}

    # Merge every MIDI track into one absolute-time stream so notes that span
    # tracks/channels are paired correctly. merge_tracks yields delta times,
    # so we accumulate them into absolute ticks as we walk.
    abs_t = 0
    for msg in mido.merge_tracks(mid.tracks):
        abs_t += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "program_change":
            programs[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            open_notes[(msg.channel, msg.note)] = (abs_t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in open_notes:
                start, vel = open_notes.pop(key)
                ch = msg.channel
                is_drum = ch == DRUM_CHANNEL
                if ch not in tracks:
                    tracks[ch] = Track(ch, programs.get(ch, 0), is_drum)
                tracks[ch].notes.append(Note(start, abs_t - start, msg.note, vel))

    # Close any notes left hanging at end of file.
    for (ch, pitch), (start, vel) in open_notes.items():
        is_drum = ch == DRUM_CHANNEL
        if ch not in tracks:
            tracks[ch] = Track(ch, programs.get(ch, 0), is_drum)
        tracks[ch].notes.append(Note(start, tpb, pitch, vel))

    ordered = []
    for ch in sorted(tracks):
        t = tracks[ch]
        t.notes.sort(key=lambda n: (n.start, n.pitch))
        t.program = programs.get(ch, 0)
        if t.notes:
            ordered.append(t)
    return ordered, tpb, tempo


# --------------------------------------------------------------------------- #
#  Probability analysis
# --------------------------------------------------------------------------- #

def dur_label(dur, tpb):
    """Map a tick duration to the nearest common note-value label."""
    beats = dur / tpb
    table = [
        (4.0, "whole"), (3.0, "dotted half"), (2.0, "half"),
        (1.5, "dotted qtr"), (1.0, "quarter"), (0.75, "dotted 8th"),
        (0.5, "eighth"), (0.375, "dotted 16th"), (0.25, "16th"),
        (0.125, "32nd"),
    ]
    best = min(table, key=lambda x: abs(math.log(max(beats, 1e-6)) - math.log(x[0])))
    return best[1]


def normalize(counter: Counter):
    """Turn a Counter into a list of (label, probability) sorted by prob."""
    total = sum(counter.values()) or 1
    items = [(k, v / total) for k, v in counter.items()]
    items.sort(key=lambda x: -x[1])
    return items


def analyze_track(track: Track, tpb: int):
    """Compute probability templates for one track.

    The returned dict carries both human-readable histograms (for charts)
    and the raw structures the remixer samples from.
    """
    notes = track.notes
    stats = {"is_drum": track.is_drum, "name": track.name, "count": len(notes)}

    if track.is_drum:
        # ---- Rhythmic / percussion statistics ----
        grid = 16  # slots per bar (16th-note resolution)
        ticks_per_bar = tpb * 4
        slot_ticks = ticks_per_bar / grid

        instrument_hits = Counter()         # which drum pieces are used
        per_slot = defaultdict(Counter)     # drum -> Counter over bar position
        slot_activity = Counter()           # overall onset position in bar
        velocities = defaultdict(list)      # drum -> velocities
        for n in notes:
            drum = GM_DRUMS.get(n.pitch, f"Note {n.pitch}")
            instrument_hits[drum] += 1
            slot = int(round((n.start % ticks_per_bar) / slot_ticks)) % grid
            per_slot[n.pitch][slot] += 1
            slot_activity[slot] += 1
            velocities[n.pitch].append(n.velocity)

        bars = max(1, int(math.ceil((track.span or ticks_per_bar) / ticks_per_bar)))
        # Onset probability per (drum, slot) = hits / bars (clamped to 1.0).
        slot_prob = {}
        for pitch, slots in per_slot.items():
            slot_prob[pitch] = {s: min(1.0, c / bars) for s, c in slots.items()}

        stats.update({
            "instrument_hist": normalize(instrument_hits),
            "slot_activity": [slot_activity.get(s, 0) for s in range(grid)],
            "grid": grid,
            "bars": bars,
            "slot_ticks": slot_ticks,
            "ticks_per_bar": ticks_per_bar,
            "_slot_prob": slot_prob,
            "_velocities": {p: v for p, v in velocities.items()},
        })
        return stats

    # ---- Melodic statistics ----
    pitch_classes = Counter()
    intervals = Counter()
    durations = Counter()
    velocities = Counter()
    transitions = defaultdict(Counter)  # pitch -> Counter(next pitch)
    starts = Counter()                  # candidate starting pitches

    prev = None
    for i, n in enumerate(notes):
        pitch_classes[PITCH_NAMES[n.pitch % 12]] += 1
        durations[dur_label(n.dur, tpb)] += 1
        velocities[(n.velocity // 16) * 16] += 1
        if i == 0:
            starts[n.pitch] += 1
        if prev is not None:
            iv = n.pitch - prev
            intervals[iv] += 1
            transitions[prev][n.pitch] += 1
        prev = n.pitch

    # Raw duration distribution in ticks (for resampling actual rhythm).
    dur_ticks = Counter(n.dur for n in notes)

    stats.update({
        "pitch_hist": normalize(pitch_classes),
        "interval_hist": _interval_hist(intervals),
        "dur_hist": normalize(durations),
        "vel_hist": [(f"{k}-{k+15}", p) for k, p in normalize(velocities)],
        "_transitions": transitions,
        "_starts": starts,
        "_pitch_pool": Counter(n.pitch for n in notes),
        "_dur_ticks": dur_ticks,
        "_velocities": Counter(n.velocity for n in notes),
    })
    return stats


def _interval_hist(intervals: Counter):
    names = {0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4",
             6: "tritone", 7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7",
             12: "octave"}

    def label(iv):
        sign = "+" if iv > 0 else "-" if iv < 0 else ""
        nm = names.get(abs(iv), f"{abs(iv)}st")
        return f"{sign}{nm}"

    folded = Counter()
    for iv, c in intervals.items():
        folded[label(iv)] += c
    return normalize(folded)


# --------------------------------------------------------------------------- #
#  Key / scale detection
# --------------------------------------------------------------------------- #

def detect_scale(tracks):
    """Return (tonic_pc, scale_pitch_classes, label) using key profiles."""
    pc_weight = [0.0] * 12
    for t in tracks:
        if t.is_drum:
            continue
        for n in t.notes:
            pc_weight[n.pitch % 12] += n.dur

    if sum(pc_weight) == 0:
        return 0, set(MAJOR_SCALE), "C major"

    best = None
    for tonic in range(12):
        for profile, mode, template in (
            (MAJOR_PROFILE, "major", MAJOR_SCALE),
            (MINOR_PROFILE, "minor", NATURAL_MINOR),
        ):
            score = sum(pc_weight[(tonic + i) % 12] * profile[i] for i in range(12))
            if best is None or score > best[0]:
                best = (score, tonic, mode, template)

    _, tonic, mode, template = best
    scale = {(tonic + i) % 12 for i in template}
    return tonic, scale, f"{PITCH_NAMES[tonic]} {mode}"


def snap_to_scale(pitch, scale):
    """Move a pitch to the nearest pitch class that is in the scale."""
    for off in range(0, 7):
        for cand in (pitch - off, pitch + off):
            if cand % 12 in scale:
                return cand
    return pitch


# --------------------------------------------------------------------------- #
#  Remix engine (sampling from the probability templates)
# --------------------------------------------------------------------------- #

def weighted_choice(counter: Counter, rng: random.Random):
    total = sum(counter.values())
    if total == 0:
        return None
    r = rng.uniform(0, total)
    upto = 0
    for k, w in counter.items():
        upto += w
        if upto >= r:
            return k
    return k


def remix_melodic(track: Track, stats: dict, rng: random.Random):
    """Generate a new melody filling the original span using the Markov
    pitch chain + duration/velocity histograms."""
    span = track.span or (stats.get("count", 8) * 240)
    origin = min((n.start for n in track.notes), default=0)
    end = origin + span

    transitions = stats["_transitions"]
    starts = stats["_starts"] or stats["_pitch_pool"]
    pool = stats["_pitch_pool"]
    dur_ticks = stats["_dur_ticks"]
    vels = stats["_velocities"]

    pitch = weighted_choice(starts, rng) or weighted_choice(pool, rng) or 60
    t = origin
    new_notes = []
    guard = 0
    while t < end and guard < 4096:
        guard += 1
        dur = weighted_choice(dur_ticks, rng) or 240
        vel = weighted_choice(vels, rng) or 80
        if t + dur > end:
            dur = max(1, end - t)
        new_notes.append(Note(t, dur, pitch, vel))
        t += dur
        nxt = weighted_choice(transitions.get(pitch, Counter()), rng)
        if nxt is None:
            nxt = weighted_choice(pool, rng)
        pitch = nxt if nxt is not None else pitch
    new = Track(track.channel, track.program, False)
    new.notes = new_notes
    return new


def remix_drums(track: Track, stats: dict, rng: random.Random):
    """Regenerate a groove by sampling each (drum, bar-slot) onset from its
    measured probability."""
    grid = stats["grid"]
    bars = stats["bars"]
    slot_ticks = stats["slot_ticks"]
    ticks_per_bar = stats["ticks_per_bar"]
    slot_prob = stats["_slot_prob"]
    velocities = stats["_velocities"]

    new = Track(track.channel, track.program, True)
    for bar in range(bars):
        bar_start = bar * ticks_per_bar
        for pitch, slots in slot_prob.items():
            vlist = velocities.get(pitch) or [90]
            for slot, p in slots.items():
                if rng.random() < p:
                    start = int(bar_start + slot * slot_ticks)
                    vel = vlist[rng.randrange(len(vlist))]
                    new.notes.append(Note(start, int(slot_ticks), pitch, vel))
    new.notes.sort(key=lambda n: n.start)
    return new


# --------------------------------------------------------------------------- #
#  Counterpoint reconciliation
# --------------------------------------------------------------------------- #

def pitch_at(notes, t):
    """Pitch sounding at time t (or nearest earlier note's pitch)."""
    sounding = None
    last = None
    for n in notes:
        if n.start <= t:
            last = n
        if n.start <= t < n.end:
            sounding = n.pitch
        if n.start > t:
            break
    if sounding is not None:
        return sounding
    return last.pitch if last else None


def apply_counterpoint(melodic_tracks, scale, rng: random.Random):
    """Reconcile generated melodic voices with species-counterpoint rules.

    * The lowest-average voice becomes the bass (cantus firmus).
    * Every other voice is snapped to the scale and nudged so each onset is
      consonant with the bass, preferring imperfect consonances and avoiding
      parallel perfect fifths/octaves with the previous onset.
    """
    if not melodic_tracks:
        return

    # First, snap everything to the detected scale.
    for tr in melodic_tracks:
        for n in tr.notes:
            n.pitch = snap_to_scale(n.pitch, scale)

    if len(melodic_tracks) == 1:
        return

    bass = min(melodic_tracks, key=lambda t: _avg_pitch(t))
    bass_notes = bass.notes

    for tr in melodic_tracks:
        if tr is bass:
            continue
        prev_interval = None
        prev_dir = 0
        prev_pitch = None
        for n in tr.notes:
            b = pitch_at(bass_notes, n.start)
            if b is None:
                continue
            n.pitch = _best_consonant(
                target=n.pitch, bass=b, scale=scale,
                prev_interval=prev_interval, prev_pitch=prev_pitch, rng=rng,
            )
            interval = (n.pitch - b) % 12
            prev_dir = 0 if prev_pitch is None else _sign(n.pitch - prev_pitch)
            prev_interval = interval
            prev_pitch = n.pitch


def _avg_pitch(track):
    return sum(n.pitch for n in track.notes) / max(1, len(track.notes))


def _sign(x):
    return (x > 0) - (x < 0)


def _best_consonant(target, bass, scale, prev_interval, prev_pitch, rng):
    """Pick the scale pitch near `target`, above `bass`, that forms the best
    consonance with the bass under counterpoint rules."""
    candidates = []
    for off in range(-7, 8):
        c = target + off
        if c % 12 not in scale:
            continue
        if c < bass:  # keep this voice at/above the bass
            continue
        interval = (c - bass) % 12
        if interval not in CONSONANCES:
            continue
        score = 0.0
        # Prefer imperfect consonances (richer than bare 5ths/octaves).
        score += 2.0 if interval in IMPERFECT_CONSONANCES else 0.0
        # Penalise distance from the Markov-suggested pitch.
        score -= 0.35 * abs(off)
        # Avoid parallel perfect fifths/octaves with the previous onset.
        if (prev_interval in PERFECT_CONSONANCES and interval == prev_interval
                and prev_pitch is not None and _sign(c - prev_pitch) != 0):
            score -= 5.0
        # Mild penalty for big melodic leaps in this voice.
        if prev_pitch is not None:
            score -= 0.05 * abs(c - prev_pitch)
        candidates.append((score, c))

    if not candidates:
        return snap_to_scale(target, scale)
    best = max(c[0] for c in candidates)
    top = [c for s, c in candidates if s >= best - 1e-9]
    return rng.choice(top)


# --------------------------------------------------------------------------- #
#  Remix orchestration
# --------------------------------------------------------------------------- #

def remix_all(tracks, tpb, tempo, seed=None):
    """Produce a fully remixed, counterpoint-reconciled set of tracks plus
    the analysis of the result."""
    rng = random.Random(seed)
    scale_info = detect_scale(tracks)
    _, scale, scale_label = scale_info

    new_tracks = []
    melodic = []
    for tr in tracks:
        stats = analyze_track(tr, tpb)
        if tr.is_drum:
            nt = remix_drums(tr, stats, rng)
        else:
            nt = remix_melodic(tr, stats, rng)
            melodic.append(nt)
        new_tracks.append(nt)

    apply_counterpoint(melodic, scale, rng)
    return new_tracks, scale_label


def tracks_to_midi(tracks, tpb, tempo):
    """Render Track objects back into MIDI bytes."""
    mid = mido.MidiFile(ticks_per_beat=tpb)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    mid.tracks.append(meta)

    for tr in tracks:
        mt = mido.MidiTrack()
        if not tr.is_drum:
            mt.append(mido.Message("program_change", channel=tr.channel,
                                   program=tr.program or 0, time=0))
        events = []
        for n in tr.notes:
            v = max(1, min(127, int(n.velocity)))
            p = max(0, min(127, int(n.pitch)))
            events.append((n.start, 1, mido.Message(
                "note_on", channel=tr.channel, note=p, velocity=v, time=0)))
            events.append((n.end, 0, mido.Message(
                "note_off", channel=tr.channel, note=p, velocity=0, time=0)))
        events.sort(key=lambda e: (e[0], e[1]))
        last = 0
        for abs_t, _, msg in events:
            msg.time = max(0, int(abs_t - last))
            last = abs_t
            mt.append(msg)
        mid.tracks.append(mt)

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  SVG chart helpers (no matplotlib needed)
# --------------------------------------------------------------------------- #

def svg_bars(title, items, color="#5b8def", max_bars=12):
    """Horizontal probability bar chart from [(label, prob), ...]."""
    items = items[:max_bars]
    if not items:
        return f"<div class='chart'><h4>{html.escape(title)}</h4>" \
               f"<p class='empty'>no data</p></div>"
    rows = []
    row_h = 20
    width = 320
    label_w = 96
    bar_w = width - label_w - 44
    mx = max(p for _, p in items) or 1
    for i, (lab, p) in enumerate(items):
        y = i * row_h
        w = max(1, bar_w * (p / mx))
        rows.append(
            f"<text x='0' y='{y+14}' class='lab'>{html.escape(str(lab))}</text>"
            f"<rect x='{label_w}' y='{y+3}' width='{w:.1f}' height='14' "
            f"rx='3' fill='{color}'></rect>"
            f"<text x='{label_w+w+4:.1f}' y='{y+14}' class='val'>"
            f"{p*100:.0f}%</text>"
        )
    h = len(items) * row_h + 6
    return (
        f"<div class='chart'><h4>{html.escape(title)}</h4>"
        f"<svg width='{width}' height='{h}' viewBox='0 0 {width} {h}'>"
        + "".join(rows) + "</svg></div>"
    )


def svg_groove(title, activity, grid):
    """Step-sequencer style chart of onset activity across the bar."""
    if not activity:
        return ""
    cell = 18
    width = grid * cell + 1
    mx = max(activity) or 1
    cells = []
    for s in range(grid):
        x = s * cell
        intensity = activity[s] / mx
        shade = int(40 + 200 * intensity)
        beat = (s % 4 == 0)
        border = "#888" if beat else "#333"
        fill = f"rgb({shade//3},{shade//2},{shade})" if activity[s] else "#1b1f2a"
        cells.append(
            f"<rect x='{x}' y='0' width='{cell-1}' height='{cell-1}' "
            f"rx='2' fill='{fill}' stroke='{border}'></rect>"
        )
    return (
        f"<div class='chart'><h4>{html.escape(title)}</h4>"
        f"<svg width='{width}' height='{cell}' viewBox='0 0 {width} {cell}'>"
        + "".join(cells) + "</svg>"
        f"<p class='hint'>16 bar positions · brighter = more frequent onset</p>"
        f"</div>"
    )


def render_track_card(stats):
    """Build the HTML card of charts for one analyzed track."""
    name = html.escape(stats["name"])
    head = f"<div class='card'><h3>{name} " \
           f"<span class='count'>{stats['count']} notes</span></h3>"
    if stats["is_drum"]:
        body = (
            svg_groove("Onset positions in bar", stats["slot_activity"], stats["grid"])
            + svg_bars("Drum piece probability", stats["instrument_hist"], "#e0795b")
        )
    else:
        body = (
            svg_bars("Pitch-class probability", stats["pitch_hist"], "#5b8def")
            + svg_bars("Melodic interval probability", stats["interval_hist"], "#7d5bef")
            + svg_bars("Note duration probability", stats["dur_hist"], "#5bbf9a")
            + svg_bars("Velocity probability", stats["vel_hist"], "#bf9a5b")
        )
    return head + "<div class='charts'>" + body + "</div></div>"


# --------------------------------------------------------------------------- #
#  Web application
# --------------------------------------------------------------------------- #

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIDI Probability Remix Studio</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;
         background:#0d1018; color:#e7eaf3; }
  header { padding:24px 28px; background:linear-gradient(120deg,#1a2236,#11192b);
           border-bottom:1px solid #222a3d; }
  h1 { margin:0 0 4px; font-size:22px; }
  header p { margin:0; color:#9aa6c4; }
  main { max-width:1180px; margin:0 auto; padding:24px 28px 80px; }
  .controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap;
              background:#141a28; border:1px solid #222a3d; border-radius:14px;
              padding:18px; margin-bottom:22px; }
  .btn { border:0; border-radius:10px; padding:11px 18px; font-size:15px;
         font-weight:600; cursor:pointer; color:#fff; }
  .btn-primary { background:#5b8def; }
  .btn-remix { background:linear-gradient(120deg,#e0795b,#bf5bef); font-size:16px; }
  .btn-remix:disabled { opacity:.4; cursor:not-allowed; }
  .btn-ghost { background:#28324a; }
  .file { color:#9aa6c4; }
  .pill { background:#1d2740; border:1px solid #2c3a5e; border-radius:999px;
          padding:6px 14px; font-size:13px; color:#bcd0ff; }
  .col2 { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
  @media (max-width:980px){ .col2 { grid-template-columns:1fr; } }
  section h2 { font-size:15px; text-transform:uppercase; letter-spacing:.08em;
               color:#8ea2cf; border-bottom:1px solid #222a3d; padding-bottom:8px; }
  .card { background:#141a28; border:1px solid #222a3d; border-radius:14px;
          padding:16px 18px; margin-bottom:18px; }
  .card h3 { margin:0 0 12px; font-size:16px; }
  .count { font-size:12px; color:#8ea2cf; font-weight:400; }
  .charts { display:flex; flex-wrap:wrap; gap:18px; }
  .chart h4 { margin:0 0 6px; font-size:12px; color:#9aa6c4; font-weight:600; }
  .chart .lab { fill:#c6cee0; font-size:11px; }
  .chart .val { fill:#7f8db0; font-size:10px; }
  .chart .empty,.hint { color:#6b768f; font-size:11px; }
  .hint { color:#6b768f; font-size:11px; margin:4px 0 0; }
  .empty-state { color:#7f8db0; padding:30px; text-align:center;
                 border:1px dashed #2c3a5e; border-radius:14px; }
  a.download { color:#7fd0ff; }
  .saved { list-style:none; margin:0 0 22px; padding:0; display:flex;
           flex-direction:column; gap:8px; }
  .saved li { display:flex; align-items:center; gap:12px; background:#141a28;
              border:1px solid #222a3d; border-radius:10px; padding:10px 14px; }
  .saved li.fresh { border-color:#bf5bef; }
  .saved .idx { color:#7f8db0; font-variant-numeric:tabular-nums; }
  .saved .fname { font-weight:600; color:#e7eaf3; word-break:break-all; }
  .saved .meta { color:#8ea2cf; font-size:13px; }
  .saved .actions { margin-left:auto; display:flex; align-items:center; gap:10px; }
  .saved a { color:#7fd0ff; white-space:nowrap; }
  .btn-mini { border:0; border-radius:8px; padding:7px 13px; font-size:13px;
              font-weight:600; cursor:pointer; color:#fff; background:#2e7d5b;
              white-space:nowrap; }
  .btn-mini.stop { background:#b5485f; }
  .spin { display:inline-block; width:16px; height:16px; border:3px solid #fff5;
          border-top-color:#fff; border-radius:50%; animation:s .8s linear infinite;
          vertical-align:-3px; }
  @keyframes s { to { transform:rotate(360deg); } }
</style></head>
<body>
<header>
  <h1>🎲 MIDI Probability Remix Studio</h1>
  <p>Upload a MIDI file → study its instrument probabilities → remix every track
     from those templates → counterpoint keeps the parts in harmony.</p>
</header>
<main>
  <div class="controls">
    <input type="file" id="file" accept=".mid,.midi" class="file">
    <button class="btn btn-primary" id="analyzeBtn">Analyze</button>
    <button class="btn btn-remix" id="remixBtn" disabled>🎲 Random Remix</button>
    <span id="status"></span>
    <span class="pill" id="keyPill" style="display:none"></span>
  </div>
  <section id="savedSection" style="display:none">
    <h2>Saved remixes <span class="count" id="savedCount"></span></h2>
    <p class="hint">▶ play streams the remix through a built-in Web Audio
       synthesizer — no plugins or internet needed.</p>
    <ul id="saved" class="saved"></ul>
  </section>
  <div class="col2">
    <section>
      <h2>Original probabilities</h2>
      <div id="orig"><div class="empty-state">Upload a .mid file and press Analyze.</div></div>
    </section>
    <section>
      <h2>Remix probabilities</h2>
      <div id="remix"><div class="empty-state">Press 🎲 Random Remix to generate a new tune.</div></div>
    </section>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let loaded = false;

function setStatus(t, spin){ $('#status').innerHTML = (spin?'<span class=spin></span> ':'')+t; }

function esc(s){ return String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function renderHistory(history){
  const ul = $('#saved');
  $('#savedSection').style.display = history.length ? 'block' : 'none';
  $('#savedCount').textContent = history.length
    ? '('+history.length+' auto-saved to ./remixes/)' : '';
  ul.innerHTML = history.map((h,i) => {
    const isPlaying = playing && playing.file===h.file;
    return '<li class="'+(i===0?'fresh':'')+'">'
    + '<span class="idx">#'+(history.length-i)+'</span>'
    + '<span><span class="fname">'+esc(h.file)+'</span><br>'
    + '<span class="meta">'+esc(h.time)+' · '+esc(h.key)+' · '
    + h.tracks+' tracks</span></span>'
    + '<span class="actions">'
    + '<button class="btn-mini play'+(isPlaying?' stop':'')+'" data-file="'
    + esc(h.file)+'">'+(isPlaying?'■ stop':'▶ play')+'</button>'
    + '<a class="download" href="'+esc(h.url)+'" download>⬇ download</a>'
    + '</span></li>';
  }).join('');
  // Re-bind the playing button reference after a re-render.
  if(playing){
    const b = ul.querySelector('.play[data-file="'+CSS.escape(playing.file)+'"]');
    playing.btn = b || playing.btn;
  }
}

async function postJSON(url, body){
  const r = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body||{})});
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

/* ---- Built-in Web Audio MIDI player (no plugins / no network) ---- */
let audioCtx = null;
let playing = null;   // { file, nodes:[], timer, btn }

function stopPlayback(){
  if(!playing) return;
  playing.nodes.forEach(n => { try { n.stop(); } catch(e){} });
  if(playing.timer) clearTimeout(playing.timer);
  if(playing.btn){ playing.btn.textContent='▶ play'; playing.btn.classList.remove('stop'); }
  playing = null;
}

function tone(pit, start, dur, amp, ti, out){
  const o = audioCtx.createOscillator();
  o.type = ['triangle','sawtooth','square','sine'][ti % 4];
  o.frequency.value = 440 * Math.pow(2, (pit-69)/12);
  const g = audioCtx.createGain();
  const peak = 0.10 + 0.45*amp, sus = Math.max(0.001, peak*0.55);
  g.gain.setValueAtTime(0.0001, start);
  g.gain.linearRampToValueAtTime(peak, start+0.012);
  g.gain.exponentialRampToValueAtTime(sus, start+Math.min(dur,0.18));
  g.gain.setValueAtTime(sus, Math.max(start+0.02, start+dur-0.05));
  g.gain.exponentialRampToValueAtTime(0.0001, start+dur);
  o.connect(g); g.connect(out);
  o.start(start); o.stop(start+dur+0.03);
  return o;
}

function drum(pit, start, amp, out){
  if(pit===35 || pit===36){            // kick
    const o=audioCtx.createOscillator(), g=audioCtx.createGain();
    o.frequency.setValueAtTime(150, start);
    o.frequency.exponentialRampToValueAtTime(50, start+0.12);
    g.gain.setValueAtTime(amp*0.9, start);
    g.gain.exponentialRampToValueAtTime(0.0001, start+0.18);
    o.connect(g); g.connect(out); o.start(start); o.stop(start+0.2);
    return o;
  }
  const isHat = pit>=42 && pit<=46;
  const dur = (pit===42||pit===44)?0.05 : (pit===46?0.3 : 0.16);
  const buf = audioCtx.createBuffer(1, Math.ceil(audioCtx.sampleRate*dur), audioCtx.sampleRate);
  const d = buf.getChannelData(0);
  for(let i=0;i<d.length;i++) d[i] = Math.random()*2-1;
  const src=audioCtx.createBufferSource(); src.buffer=buf;
  const f=audioCtx.createBiquadFilter();
  f.type = isHat ? 'highpass' : 'bandpass';
  f.frequency.value = isHat ? 8000 : ((pit===38||pit===40)?1800:3000);
  const g=audioCtx.createGain();
  g.gain.setValueAtTime(amp*0.6, start);
  g.gain.exponentialRampToValueAtTime(0.0001, start+dur);
  src.connect(f); f.connect(g); g.connect(out);
  src.start(start); src.stop(start+dur);
  return src;
}

async function play(file, btn){
  if(playing && playing.file===file){ stopPlayback(); return; }
  stopPlayback();
  let data;
  try{ data = await (await fetch('/events/'+encodeURIComponent(file))).json(); }
  catch(e){ setStatus('Could not load remix audio: '+e.message); return; }
  if(!audioCtx) audioCtx = new (window.AudioContext||window.webkitAudioContext)();
  await audioCtx.resume();
  const secPerTick = (data.tempo/1e6) / data.tpb;
  const t0 = audioCtx.currentTime + 0.08;
  const comp = audioCtx.createDynamicsCompressor(); comp.connect(audioCtx.destination);
  const master = audioCtx.createGain(); master.gain.value = 0.6; master.connect(comp);
  const nodes = []; let end = t0;
  data.tracks.forEach((tr, ti) => {
    tr.notes.forEach(n => {
      const start = t0 + n[0]*secPerTick;
      const dur = Math.max(0.05, n[1]*secPerTick);
      const amp = n[3]/127;
      nodes.push(tr.is_drum ? drum(n[2], start, amp, master)
                            : tone(n[2], start, dur, amp, ti, master));
      end = Math.max(end, start+dur);
    });
  });
  playing = { file, nodes, btn, timer:null };
  if(btn){ btn.textContent='■ stop'; btn.classList.add('stop'); }
  playing.timer = setTimeout(stopPlayback, (end - audioCtx.currentTime + 0.4)*1000);
}

$('#saved').addEventListener('click', e => {
  const b = e.target.closest('.play');
  if(b) play(b.dataset.file, b);
});

$('#analyzeBtn').onclick = async () => {
  const f = $('#file').files[0];
  if(!f){ setStatus('Choose a .mid file first.'); return; }
  setStatus('Analyzing…', true);
  const buf = await f.arrayBuffer();
  let bin=''; const bytes=new Uint8Array(buf);
  for(let i=0;i<bytes.length;i++) bin+=String.fromCharCode(bytes[i]);
  try{
    const res = await postJSON('/analyze',{name:f.name, data:btoa(bin)});
    $('#orig').innerHTML = res.html;
    $('#keyPill').style.display='inline-block';
    $('#keyPill').textContent = 'Detected key: '+res.key;
    $('#remixBtn').disabled = false;
    loaded = true;
    renderHistory(res.history || []);
    setStatus('Analyzed '+res.tracks+' instrument track(s).');
  }catch(e){ setStatus('Error: '+e.message); }
};

$('#remixBtn').onclick = async () => {
  if(!loaded) return;
  setStatus('Remixing & applying counterpoint…', true);
  try{
    const res = await postJSON('/remix',{});
    $('#remix').innerHTML = res.html;
    renderHistory(res.history || []);
    $('#keyPill').textContent = 'Key: '+res.key+' · counterpoint applied';
    setStatus('Remix saved as '+res.saved.file+' — '+res.tracks+' track(s).');
  }catch(e){ setStatus('Error: '+e.message); }
};
</script>
</body></html>"""


def _safe_stem(name):
    """Turn an uploaded filename into a safe filename stem."""
    stem = os.path.splitext(os.path.basename(name or "song"))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "song"
    return stem[:40]


def save_remix(stem, data: bytes):
    """Write remix bytes to REMIX_DIR under a unique, timestamped filename.

    Returns the bare filename (served later from /remixes/<filename>).
    """
    os.makedirs(REMIX_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{stem}_remix_{ts}"
    fname = base + ".mid"
    # Guard against same-second collisions.
    n = 2
    while os.path.exists(os.path.join(REMIX_DIR, fname)):
        fname = f"{base}_{n}.mid"
        n += 1
    with open(os.path.join(REMIX_DIR, fname), "wb") as fh:
        fh.write(data)
    return fname


class Studio:
    """Holds the currently loaded song + the history of saved remixes."""

    def __init__(self):
        self.tracks = None
        self.tpb = 480
        self.tempo = 500000
        self.source_name = "song"
        self.history = []  # list of {file, key, tracks, time}


STUDIO = Studio()


class Handler(BaseHTTPRequestHandler):
    server_version = "MidiRemixStudio/1.0"

    def log_message(self, *args):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path.startswith("/index"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/history":
            self._send(200, json.dumps({"history": STUDIO.history}))
        elif path.startswith("/remixes/"):
            self._serve_remix_file(path[len("/remixes/"):])
        elif path.startswith("/events/"):
            self._serve_events(path[len("/events/"):])
        else:
            self._send(404, "Not found", "text/plain")

    def _safe_remix_path(self, fname):
        """Validate a bare remix filename and return its full path, else None."""
        if not fname or "/" in fname or "\\" in fname or ".." in fname:
            return None
        full = os.path.join(REMIX_DIR, fname)
        return full if os.path.isfile(full) else None

    def _serve_remix_file(self, fname):
        full = self._safe_remix_path(fname)
        if full is None:
            self._send(404, "No such remix", "text/plain")
            return
        with open(full, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", "audio/midi")
        self.send_header("Content-Disposition",
                         f'attachment; filename="{fname}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self, fname):
        """Return a saved remix as JSON note events for the in-browser player."""
        full = self._safe_remix_path(fname)
        if full is None:
            self._send(404, "No such remix", "text/plain")
            return
        with open(full, "rb") as fh:
            tracks, tpb, tempo = load_tracks(fh.read())
        out = {"tpb": tpb, "tempo": tempo, "tracks": [
            {"is_drum": t.is_drum, "program": t.program, "channel": t.channel,
             "notes": [[n.start, n.dur, n.pitch, n.velocity] for n in t.notes]}
            for t in tracks
        ]}
        self._send(200, json.dumps(out))

    def do_POST(self):
        try:
            if self.path == "/analyze":
                self._handle_analyze()
            elif self.path == "/remix":
                self._handle_remix()
            else:
                self._send(404, "Not found", "text/plain")
        except Exception as exc:  # surface errors to the UI
            self._send(500, f"{type(exc).__name__}: {exc}", "text/plain")

    def _handle_analyze(self):
        payload = self._read_json()
        data = base64.b64decode(payload["data"])
        tracks, tpb, tempo = load_tracks(data)
        if not tracks:
            raise ValueError("No note data found in this MIDI file.")
        STUDIO.tracks, STUDIO.tpb, STUDIO.tempo = tracks, tpb, tempo
        STUDIO.source_name = _safe_stem(payload.get("name"))
        STUDIO.history = []  # fresh remix history for the newly loaded song
        _, _, key = detect_scale(tracks)
        cards = "".join(render_track_card(analyze_track(t, tpb)) for t in tracks)
        self._send(200, json.dumps({"html": cards, "key": key,
                                    "tracks": len(tracks),
                                    "history": STUDIO.history}))

    def _handle_remix(self):
        if not STUDIO.tracks:
            raise ValueError("Upload and analyze a MIDI file first.")
        new_tracks, key = remix_all(
            STUDIO.tracks, STUDIO.tpb, STUDIO.tempo, seed=random.randrange(1 << 30))
        data = tracks_to_midi(new_tracks, STUDIO.tpb, STUDIO.tempo)
        fname = save_remix(STUDIO.source_name, data)
        entry = {
            "file": fname,
            "url": "/remixes/" + fname,
            "key": key,
            "tracks": len(new_tracks),
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
        }
        STUDIO.history.insert(0, entry)  # newest first
        cards = "".join(render_track_card(analyze_track(t, STUDIO.tpb))
                        for t in new_tracks)
        self._send(200, json.dumps({"html": cards, "key": key,
                                    "tracks": len(new_tracks),
                                    "history": STUDIO.history,
                                    "saved": entry}))


def serve(port=8765, open_browser=True):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"MIDI Probability Remix Studio running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.server_close()


# --------------------------------------------------------------------------- #
#  Headless self-test (no browser, no GUI) — builds a tune, remixes it.
# --------------------------------------------------------------------------- #

def _build_demo_midi():
    """Create a 2-bar, 3-track (lead, bass, drums) demo MIDI in memory."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tpb = 480

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    mid.tracks.append(meta)

    # Lead melody on channel 0 (C major-ish).
    lead = mido.MidiTrack()
    lead.append(mido.Message("program_change", channel=0, program=0, time=0))
    for p in [60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 67, 72, 67, 64, 60, 60]:
        lead.append(mido.Message("note_on", channel=0, note=p, velocity=80, time=0))
        lead.append(mido.Message("note_off", channel=0, note=p, velocity=0, time=tpb // 2))
    mid.tracks.append(lead)

    # Bass on channel 1.
    bass = mido.MidiTrack()
    bass.append(mido.Message("program_change", channel=1, program=33, time=0))
    for p in [36, 43, 41, 38, 36, 43, 41, 38]:
        bass.append(mido.Message("note_on", channel=1, note=p, velocity=90, time=0))
        bass.append(mido.Message("note_off", channel=1, note=p, velocity=0, time=tpb))
    mid.tracks.append(bass)

    # Drums on channel 9.
    drums = mido.MidiTrack()
    pattern = [(36, 0), (42, 0), (42, 240), (38, 480), (42, 480),
               (42, 720), (36, 960), (42, 960), (42, 1200),
               (38, 1440), (42, 1440), (42, 1680)]
    events = []
    for note, t in pattern:
        events.append((t, "on", note))
        events.append((t + 100, "off", note))
    events.sort(key=lambda e: e[0])
    last = 0
    for t, kind, note in events:
        dt = t - last
        last = t
        if kind == "on":
            drums.append(mido.Message("note_on", channel=9, note=note, velocity=100, time=dt))
        else:
            drums.append(mido.Message("note_off", channel=9, note=note, velocity=0, time=dt))
    mid.tracks.append(drums)

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def selftest():
    print("Building demo MIDI…")
    data = _build_demo_midi()
    tracks, tpb, tempo = load_tracks(data)
    print(f"Parsed {len(tracks)} tracks, ticks/beat={tpb}, tempo={tempo}")
    for t in tracks:
        stats = analyze_track(t, tpb)
        kind = "drums" if t.is_drum else "melodic"
        print(f"  - {t.name}: {len(t.notes)} notes ({kind})")
        if not t.is_drum:
            top = stats["pitch_hist"][:3]
            print("      top pitch classes:",
                  ", ".join(f"{n}={p*100:.0f}%" for n, p in top))
        else:
            print("      drums used:",
                  ", ".join(f"{n}" for n, _ in stats["instrument_hist"][:4]))

    _, scale, key = detect_scale(tracks)
    print(f"Detected key: {key}")

    new_tracks, key2 = remix_all(tracks, tpb, tempo, seed=42)
    out = tracks_to_midi(new_tracks, tpb, tempo)
    assert out[:4] == b"MThd", "output is not a valid MIDI file"

    # Verify counterpoint produced consonances against the bass.
    melodic = [t for t in new_tracks if not t.is_drum]
    if len(melodic) >= 2:
        bass = min(melodic, key=_avg_pitch)
        others = [t for t in melodic if t is not bass]
        total = consonant = 0
        for tr in others:
            for n in tr.notes:
                b = pitch_at(bass.notes, n.start)
                if b is None:
                    continue
                total += 1
                if (n.pitch - b) % 12 in CONSONANCES:
                    consonant += 1
        pct = 100 * consonant / max(1, total)
        print(f"Counterpoint check: {consonant}/{total} "
              f"onsets consonant with bass ({pct:.0f}%)")
        assert pct >= 90, "counterpoint failed to enforce consonance"

    # Verify every melodic note is in the detected scale.
    in_scale = all(n.pitch % 12 in scale
                   for t in melodic for n in t.notes)
    assert in_scale, "remix produced out-of-scale notes"

    with open("remix_demo.mid", "wb") as fh:
        fh.write(out)
    print(f"Wrote remix_demo.mid ({len(out)} bytes). Self-test passed ✔")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="MIDI Probability Remix Studio")
    ap.add_argument("--port", type=int, default=8765, help="web server port")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open a browser")
    ap.add_argument("--selftest", action="store_true",
                    help="run a headless engine test and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
