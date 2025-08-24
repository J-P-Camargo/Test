Tests to detect ρ(t):
1. Control (ρ≈0): Play a steady tone (e.g., 3 kHz, 10 s). Your code should show a stable track and ρ(t) close to zero (only noise fluctuations).
2. Up-chirp (ρ>0): Play a linear chirp from 3→6 kHz in 10 s. The spectrogram will “slope” upward; the ρ(t) series should be positive throughout most of the interval (except edges).
3. Down-chirp (ρ<0): Play a chirp from 6→3 kHz in 10 s. The spectrogram slopes downward; ρ(t) becomes negative.
4. Up-then-down (ρ alternates sign): First half 3→6 kHz, then 6→3 kHz. You should see ρ(t) > 0 in the first half and ρ(t) < 0 in the second.
5. Quasi-steady (beats; ρ≈0): Two closely spaced tones (4 kHz and 4,050 Hz) with a slight 1 Hz drift over 10 s. This produces a clear beat, but without global tilt: ρ(t) should be close to zero (serves as a realistic "negative").

How to play the audio:
1. If your OS allows it, enable audio loopback (on Windows, WASAPI loopback; on macOS, use an "Aggregate/Loopback Device"). Your script already "hears" what it plays.
2. Alternative: Play through the speakers near the microphone (not ideal, but works for testing).

What to look for:
3. ρ(t) signal: positive on the ascending chirp, negative on the descending chirp, ~0 on stationary and beats.
