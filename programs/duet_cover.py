"""Duet cover weaving — pure audio logic, no ML dependencies.

Turns two RVC-converted vocal takes into a single duet vocal track,
weights.gg style. Kept free of torch so the web API (which must boot
without the ML stack) can import the mode list for input validation.

Modes:
* ``alternate``  — the two voices swap every ``duet_interval`` seconds
  (voice A sings 0..N, voice B N..2N, voice A again, ...).
* ``sequential`` — voice A sings the first ``duet_split`` percent of the
  song, voice B takes over for the rest (classic "A then B" duet).
* ``overlap``    — both voices are mixed together across the whole song
  (choir / harmony style).
"""

from pydub import AudioSegment

DUET_MODES = ("alternate", "sequential", "overlap")


def build_duet_vocals(voice_a_path, voice_b_path, output_path, duet_mode="alternate",
                      duet_split=50, duet_interval=15, crossfade_ms=80):
    """Weave two RVC-converted vocal takes into one duet vocal track.

    Every segment boundary gets a short equal-power crossfade so the voice
    switch does not click. Both inputs come from the same RVC pipeline, so
    they already share the sample rate; they are trimmed to the shorter
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
            marks = [int(duration * split / 100)]
            interval_ms = 0
        else:  # alternate
            interval_ms = int(float(duet_interval) * 1000)
            if interval_ms < 1000:
                raise Exception("The duet interval must be at least 1 second.")
            marks = list(range(interval_ms, duration, interval_ms))

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
        duet = segments[0]
        for segment in segments[1:]:
            duet = duet.append(segment, crossfade=crossfade)

    duet.export(output_path, format="wav")
    return output_path
