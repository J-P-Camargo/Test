# ========================
# Test_rho
# ========================

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import queue, time, threading
import csv

# ========================
# Main settings
# ========================
FS = 48_000
BLOCK = 4096
BAND_MIN = 500
BAND_MAX = 18_000
ENERGY_THRESH = 1e-5
PEAK_THRESH = 6.0
MAX_TRACKS = 10
TOL_HZ = FS / BLOCK
TIMEOUT_BLOCKS = 20
SMOOTH = 8
HIST_LEN = 50
SPEC_COLS = 120
F_REF = 1.0

# ========================
# Global state
# ========================
q_audio = queue.Queue()
tracks = {}
bins_f = np.fft.rfftfreq(BLOCK, d=1/FS)
band_mask = (bins_f >= BAND_MIN) & (bins_f <= BAND_MAX)
band_bins = np.where(band_mask)[0]
band_freqs = bins_f[band_mask]
n_bins = band_freqs.size
spec_hist = np.zeros((n_bins, SPEC_COLS), dtype=np.float32)

# ========================
# results buffer rho(t)
# ========================

rho_values = []
time_axis = []

# ========================
# Utilities
# ========================
def principal_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def audio_cb(indata, frames, time_info, status):
    if status:
        pass
    q_audio.put(indata.copy())

def match_track(freq):
    if not tracks:
        return None
    best_f, best_err = None, float('inf')
    for f0 in tracks.keys():
        err = abs(f0 - freq)
        if err < best_err:
            best_f, best_err = f0, err
    return best_f if best_err <= TOL_HZ else None

def ensure_track(freq):
    if freq in tracks:
        return freq
    if len(tracks) >= MAX_TRACKS:
        return None
    tracks[freq] = {
        'f0': freq,
        'prev_phase': None,
        'finst_hist': deque(maxlen=SMOOTH),
        'mag_hist': deque(maxlen=SMOOTH),
        'history': deque(maxlen=HIST_LEN),
        'miss_count': 0,
        'seen': False,
    }
    return freq

# ========================
# Block processing
# ========================
def process_block(block, n0, t_now):
    global spec_hist

    x = block.astype(np.float64)
    energy = np.mean(x**2)
    if energy < ENERGY_THRESH:
        push_spectrogram(np.zeros_like(band_freqs))
        for f0 in list(tracks):
            tracks[f0]['seen'] = False
        handle_timeouts()
        return

    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    mag = np.abs(X)

    mag_band = mag[band_mask]
    noise_floor = np.median(mag_band) + 1e-12
    norm_band = mag_band / noise_floor
    push_spectrogram(norm_band)

    peak_idx = np.where(norm_band > PEAK_THRESH)[0]
    peak_bins = band_bins[peak_idx]
    peak_freqs = bins_f[peak_bins]

    for f0 in list(tracks):
        tracks[f0]['seen'] = False

    for f in peak_freqs:
        match = match_track(f)
        if match is None:
            match = ensure_track(f)
        if match is None:
            continue
        st = tracks.pop(match) if match != f else tracks[match]
        st['f0'] = (0.9 * st['f0'] + 0.1 * f)
        key = st['f0']

        n = np.arange(len(x)) + n0
        lo = np.exp(-1j * 2 * np.pi * st['f0'] * (n / FS))
        z = np.vdot(lo, x)
        mag_c = np.abs(z) / len(x)
        phase = np.angle(z)

        dphi = 0.0 if st['prev_phase'] is None else principal_angle(phase - st['prev_phase'])
        dt = len(x) / FS
        f_dev = (dphi / (2 * np.pi)) / dt

        st['prev_phase'] = phase
        st['finst_hist'].append(st['f0'] + f_dev)
        st['mag_hist'].append(float(mag_c))
        st['history'].append(1.0)
        st['miss_count'] = 0
        st['seen'] = True

        tracks[key] = st

    handle_timeouts(seen_update=True)

    # ========================
    # ===== calculation of rho(t) =====
    # ========================     
    rho_vals = []
    for st in tracks.values():
        if len(st['finst_hist']) >= 2:
        # tempo em segundos (cada ponto = duração de um bloco)
           dt = BLOCK / FS
           tau = np.arange(len(st['finst_hist'])) * dt
           omega = np.array(st['finst_hist'])

        # covariância e variâncias consistentes (populacionais, ddof=0)
           cov = np.cov(tau, omega, ddof=0)[0, 1]
           var_tau = np.var(tau, ddof=0)
           var_omega = np.var(omega, ddof=0)

           if var_tau > 0 and var_omega > 0:
               rho = cov / np.sqrt(var_tau * var_omega)
               rho_vals.append(rho)

# average of valid tracks at this time
    if rho_vals:
        rho_mean = np.mean(rho_vals)
        rho_values.append(rho_mean)
        time_axis.append(t_now)
    

def handle_timeouts(seen_update=False):
    remove_keys = []
    for f0, st in tracks.items():
        if not st['seen']:
            st['miss_count'] += 1
            if seen_update:
                st['history'].append(0.0)
            if st['miss_count'] >= TIMEOUT_BLOCKS:
                remove_keys.append(f0)
        else:
            st['miss_count'] = 0
    for k in remove_keys:
        tracks.pop(k, None)

def push_spectrogram(col_values):
    global spec_hist
    spec_hist = np.roll(spec_hist, -1, axis=1)
    col = np.clip(np.log1p(col_values), 0, None)
    p95 = np.percentile(col, 95) + 1e-9
    spec_hist[:, -1] = np.clip(col / p95, 0.0, 1.0)

# ========================
# Visualization
# ========================
def setup_figure():
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 1, 0.05], hspace=0.3)

    ax_radar = fig.add_subplot(gs[0, 0])
    radar_img = ax_radar.imshow(np.zeros((1, HIST_LEN)), aspect='auto', cmap='Greens',
                                interpolation='nearest', vmin=0.0, vmax=1.0)
    ax_radar.set_title("Radar de portadoras")
    ax_radar.set_ylabel("Frequência (Hz)")
    ax_radar.set_xticks([])

    ax_spec = fig.add_subplot(gs[1, 0])
    spec_img = ax_spec.imshow(spec_hist, aspect='auto', origin='lower', cmap='plasma',
                              vmin=0.0, vmax=1.0, extent=[-SPEC_COLS, 0, BAND_MIN, BAND_MAX])
    ax_spec.set_title("Espectrograma (últimos blocos)")
    ax_spec.set_xlabel("Blocos")
    ax_spec.set_ylabel("Frequência (Hz)")

    ax_rho = fig.add_subplot(gs[2, 0])
    line_rho, = ax_rho.plot([], [], lw=2, color='red')
    ax_rho.set_title("Tilt espectral ρ(t)")
    ax_rho.set_xlabel("Tempo (s)")
    ax_rho.set_ylabel("ρ (t)")
    ax_rho.grid(True)

    cbar = plt.colorbar(spec_img, cax=fig.add_subplot(gs[3, 0]), orientation='horizontal')
    cbar.set_label("Intensidade normalizada")

    return fig, ax_radar, radar_img, ax_spec, spec_img, ax_rho, line_rho

def update_visual(frame, ax_radar, radar_img, ax_spec, spec_img, ax_rho, line_rho):
    spec_img.set_data(spec_hist)
    spec_img.set_extent([-SPEC_COLS, 0, BAND_MIN, BAND_MAX])

    if tracks:
        sorted_items = sorted(tracks.items(), key=lambda kv: kv[0])
        freqs = [f0 for f0, _ in sorted_items]
        hist_matrix = []
        for _, st in sorted_items:
            row = np.array(st['history'], dtype=np.float32)
            if row.size < HIST_LEN:
                row = np.pad(row, (HIST_LEN - row.size, 0))
            hist_matrix.append(row)
        radar = np.vstack(hist_matrix)
        radar_img.set_data(radar)
        ax_radar.set_yticks(np.arange(len(freqs)))
        ax_radar.set_yticklabels([f"{f0:0.1f}" for f0 in freqs])
    else:
        radar_img.set_data(np.zeros((1, HIST_LEN)))
        ax_radar.set_yticks([])
        ax_radar.set_yticklabels([])

    if rho_values:
        line_rho.set_data(time_axis, rho_values)
        ax_rho.set_xlim(max(0, time_axis[0]), time_axis[-1])
        ax_rho.set_ylim(np.min(rho_values) - 1, np.max(rho_values) + 1)

    return [radar_img, spec_img, line_rho]

# ========================
# Main loop
# ========================
def main():
    print("Captura ao vivo iniciada — fale, toque tons ou sinais; observe radar, espectrograma e ρ(t).")
    fig, ax_radar, radar_img, ax_spec, spec_img, ax_rho, line_rho = setup_figure()

    stream = sd.InputStream(channels=1, samplerate=FS, blocksize=BLOCK, dtype='float32', callback=audio_cb)
    stream.start()

    def consume():
        n0 = 0
        t0 = time.time()
        while plt.fignum_exists(fig.number):
            try:
                block = q_audio.get(timeout=0.05)[:, 0]
                process_block(block, n0, time.time()-t0)
                n0 += len(block)
            except queue.Empty:
                pass
            except Exception as e:
                pass
            time.sleep(0.001)

    t_proc = threading.Thread(target=consume, daemon=True)
    t_proc.start()

    ani = FuncAnimation(fig, update_visual,
                        fargs=(ax_radar, radar_img, ax_spec, spec_img, ax_rho, line_rho),
                        interval=100, blit=False)

    try:
        plt.show()
    finally:
        stream.stop()
        stream.close()
        # salva resultados em CSV
        with open("rho_results.csv", "w", newline="") as f: #insert the file path
            writer = csv.writer(f)
            writer.writerow(["tempo_s", "rho_t"])
            writer.writerows(zip(time_axis, rho_values))
        print("Resultados de ρ(t) salvos em rho_results.csv")

if __name__ == "__main__":
    main()
