"""Duet cover weaving — pure audio logic, no ML dependencies.

Turns two RVC-converted vocal takes into a single duet vocal track, the
way the classic weights.gg-era duet covers work. Kept free of torch so
the web API (which must boot without the ML stack) can import the mode
list for input validation.

How real duet covers share the song (research note: weights.gg itself
shut down in March 2026 and is excluded from web archives, so the
mechanism was verified against the surviving platform of that era —
Jammable, whose duet guide documents the behavior openly):

* The vocals are converted once per voice model, so both takes stay
  aligned with the original lead vocal, and are then woven together.
* Voice changes land on *vocal phrase boundaries* (the natural pauses
  between sung lines), never mid-word. "Auto pick" assigns the detected
  phrases to the voices in turn.
* Parts shorter than ~2.5 seconds are merged into the neighbouring part.
* Each voice can be pitched independently (handled by the pipeline via
  ``duet_pitch`` on the second conversion, not here).

Modes:
* ``auto``       — vocal phrases are detected on the converted vocal and
  assigned to the voices in turn (A, B, A, B …). If the take has no
  usable phrase structure (singing without pauses), it falls back to
  timed swaps every ``duet_interval`` seconds so both voices get a turn.
* ``alternate``  — timed swaps every ``duet_interval`` seconds, but each
  swap snaps to the nearest phrase boundary (within 5 s) so the switch
  happens in a pause instead of mid-word.
* ``sequential`` — voice A sings the first ``duet_split`` percent of the
  song, voice B the rest; the swap snaps to the nearest phrase boundary.
* ``overlap``    — both voices are mixed together across the whole song
  (choir / double-track effect).
"""

import numpy as np
from pydub import AudioSegment

DUET_MODES = ("auto", "alternate", "sequential", "overlap")

# RMS thresholds (dBFS) deciding when singing starts and stops. The
# hysteresis (open above OPEN_DB, close below CLOSE_DB) keeps one noisy
# phrase from flickering into many parts.
OPEN_DB = -38.0
CLOSE_DB = -45.0
# Envelope resolution.
HOP_MS = 50
# Two singing regions separated by less than this count as one phrase
# (a breath between lines does not create a voice swap).
MIN_GAP_MS = 350
# A detected singing region shorter than this is merged into its
# neighbour — the same "minimum part" rule the web duet editors use.
MIN_PHRASE_MS = 2500
# Timed swaps (alternate mode) may drift at most this far to land on a
# phrase boundary; beyond that they stay on the wall-clock mark.
MAX_SNAP_DRIFT_MS = 5000


def detect_vocal_phrases(voice, min_phrase_ms=MIN_PHRASE_MS, min_gap_ms=MIN_GAP_MS):
    """Split a vocal take into the phrases the duet voices can swap on.

    Returns a list of ``(start_ms, end_ms)`` parts that covers the whole
    segment: every instrumental gap is split evenly between the phrases
    before and after it, so weaving never drops audio. Returns ``[]``
    when no singing (only silence) is detected.
    """
    duration = len(voice)
    samples = np.array(voice.get_array_of_samples(), dtype=np.float64)
    if voice.channels > 1:
        samples = samples.reshape(-1, voice.channels).mean(axis=1)
    hop = max(1, int(voice.frame_rate * HOP_MS / 1000))
    frames = len(samples) // hop
    if frames == 0:
        return []
    rms = np.sqrt((samples[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    # Normalize by the file's integer range so dBFS is sample-width agnostic
    # (16-bit PCM -> 32768, 32-bit -> 2^31, ...).
    full_scale = float(1 << (8 * voice.sample_width - 1))
    db = 20.0 * np.log10(np.maximum(rms / full_scale, 1e-10))

    # Hysteresis pass: a phrase opens when the envelope gets loud and
    # closes only when it drops below the quieter close threshold.
    regions = []
    singing = False
    start = 0
    for index, level in enumerate(db):
        if not singing and level > OPEN_DB:
            singing = True
            start = index * HOP_MS
        elif singing and level < CLOSE_DB:
            singing = False
            regions.append([start, index * HOP_MS])
    if singing:
        regions.append([start, duration])

    if not regions:
        return []

    # Merge regions separated by a short breath...
    merged = [regions[0]]
    for region in regions[1:]:
        if region[0] - merged[-1][1] < min_gap_ms:
            merged[-1][1] = region[1]
        else:
            merged.append(region)
    # ...then absorb phrases too short to be their own part (merge into
    # the previous one; a short first phrase absorbs into the next).
    parts = []
    for region in merged:
        if region[1] - region[0] < min_phrase_ms and parts:
            parts[-1][1] = region[1]
        else:
            parts.append(region)
    if len(parts) > 1 and parts[0][1] - parts[0][0] < min_phrase_ms:
        parts[1][0] = parts[0][0]
        parts.pop(0)

    # Cover the whole timeline: every instrumental gap is split evenly
    # between the neighbouring phrases, so part i runs from the middle of
    # the gap before it to the middle of the gap after it and no audio is
    # left unassigned.
    edges = [0]
    for index in range(1, len(parts)):
        edges.append((parts[index - 1][1] + parts[index][0]) // 2)
    edges.append(duration)
    return [(edges[index], edges[index + 1]) for index in range(len(parts))]


def _snap(mark, boundaries, max_drift_ms):
    """Move ``mark`` to the nearest phrase boundary within ``max_drift_ms``."""
    if not boundaries:
        return mark
    nearest = min(boundaries, key=lambda boundary: abs(boundary - mark))
    return nearest if abs(nearest - mark) <= max_drift_ms else mark


def build_duet_vocals(voice_a_path, voice_b_path, output_path, duet_mode="auto",
                      duet_split=50, duet_interval=15, crossfade_ms=80):
    """Weave two RVC-converted vocal takes into one duet vocal track.

    Every voice change gets a short equal-power crossfade so the switch
    does not click. Both inputs come from the same RVC pipeline, so they
    already share the sample rate; they are trimmed to the shorter
    length because conversion can shift the duration by a few samples.
    """
    if duet_mode not in DUET_MODES:
        raise Exception(f'Unknown duet mode "{duet_mode}". Choose from: {", ".join(DUET_MODES)}.')
    voice_a = AudioSegment.from_wav(voice_a_path)
    voice_b = AudioSegment.from_wav(voice_b_path)
    duration = min(len(voice_a), len(voice_b))
    if duration <= 0:
        raise Exception("The converted vocals are empty — nothing to build a duet from.")
    voice_a = voice_a[:duration]
    voice_b = voice_b[:duration]

    if duet_mode == "overlap":
        duet = voice_a.overlay(voice_b)
    else:
        if duet_mode == "sequential":
            split = float(duet_split)
            if not 1 <= split <= 99:
                raise Exception("The duet split point must be between 1 and 99 percent.")
            interval_ms = 0
            marks = [int(duration * split / 100)]
        elif duet_mode == "alternate":
            interval_ms = int(float(duet_interval) * 1000)
            if interval_ms < 1000:
                raise Exception("The duet interval must be at least 1 second.")
            marks = list(range(interval_ms, duration, interval_ms))
        else:  # auto — marks come from the detected phrases below
            interval_ms = int(float(duet_interval) * 1000)
            if interval_ms < 1000:
                raise Exception("The duet interval must be at least 1 second.")
            marks = []

        phrases = detect_vocal_phrases(voice_a)
        boundaries = [end for _, end in phrases[:-1]]

        if duet_mode == "auto":
            if len(phrases) >= 2:
                # One voice per detected phrase, in turn (A, B, A, ...).
                marks = [end for _, end in phrases[:-1]]
            else:
                # No usable phrase structure (continuous singing): fall
                # back to timed swaps so both voices still get a turn.
                marks = list(range(min(interval_ms, duration), duration, interval_ms))
        elif duet_mode == "sequential":
            if boundaries:
                # Always snap: a phrase-aligned swap beats an exact % mark.
                marks = [_snap(marks[0], boundaries, duration)]
        else:  # alternate — land the timed swaps on phrase pauses
            snapped = [_snap(mark, boundaries, MAX_SNAP_DRIFT_MS) for mark in marks]
            snapped.sort()
            marks = [mark for index, mark in enumerate(snapped) if index == 0 or mark > snapped[index - 1]]

        # pydub's append(crossfade=...) overlaps the fade-out of the previous
        # segment with the fade-in of the next one (equal-power crossfade).
        crossfade = crossfade_ms
        if interval_ms:
            crossfade = min(crossfade_ms, max(1, interval_ms // 4))
        segments = []
        cursor = 0
        turn = 0  # voice A starts
        for mark in marks:
            segments.append((voice_a if turn == 0 else voice_b)[cursor:mark])
            cursor = mark
            turn = 1 - turn
        segments.append((voice_a if turn == 0 else voice_b)[cursor:])
        # Never crossfade longer than a quarter of the shortest segment.
        shortest = min(len(segment) for segment in segments)
        crossfade = min(crossfade, max(1, shortest // 4))
        duet = segments[0]
        for segment in segments[1:]:
            duet = duet.append(segment, crossfade=crossfade)

    duet.export(output_path, format="wav")
    return output_path
