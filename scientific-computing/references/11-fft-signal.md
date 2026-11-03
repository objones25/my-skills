# FFT and Signal Processing

`scipy.fft` and `scipy.signal`: spectra, filtering, convolution, and the conventions that
make the numbers come out right. The recurring failure here is not a wrong function, it is a
wrong normalization or a wrong frequency axis, both of which produce plots that look
plausible and are off by a factor of `N` or `2`.

> Timings and speed ratios in this file: date and machine not recorded. They vary by machine.

Use `scipy.fft`, not `numpy.fft`. Same conventions, plus `workers=` for threading,
`next_fast_len`, DCT/DST, and better real-input transforms.

## Real-input transforms and the frequency axis

For real-valued data the spectrum is conjugate-symmetric, so half of `fft`'s output is
redundant. `rfft` computes only the non-redundant half and `rfftfreq` gives the matching
frequencies. Always pair them.

```python
import numpy as np
from scipy import fft, signal

fs = 1000.0                      # sample rate, Hz
N = 1000
t = np.arange(N) / fs
x = 3*np.sin(2*np.pi*50*t) + 1.5*np.sin(2*np.pi*120*t)

X = fft.rfft(x)                  # (1000,) -> (501,) complex
f = fft.rfftfreq(N, d=1/fs)      # (501,)  Hz

amp = 2*np.abs(X)/N              # single-sided amplitude
amp[0] /= 2                      # DC bin is not doubled
```

```
x (1000,) -> rfft (501,)  rfftfreq (501,)  df = 1.0 Hz
peaks at [ 50. 120.]  amplitude [3.   1.5]   (true 3.0 and 1.5)
```

Three conventions to internalize:

- **`d` is the sample spacing, not the rate.** `rfftfreq(N, d=1/fs)`. Passing `fs` gives a
  frequency axis wrong by a factor of `fs**2`.
- **Bin spacing is `fs/N = 1/T`,** where `T` is the total record length. Resolution comes
  from recording longer, not from sampling faster. Zero-padding to a longer FFT interpolates
  the spectrum; it does not add resolution.
- **To get physical amplitude, scale by `2/N`** and undo the doubling on the DC bin (and, for
  even `N`, on the Nyquist bin `X[-1]`, which also has no conjugate partner).

`norm=` controls where the `1/N` lives: `'backward'` (default, `1/N` on the inverse only),
`'ortho'` (`1/sqrt(N)` on both, which makes the transform unitary and Parseval exact without
extra factors), `'forward'` (`1/N` on the forward). Pick `'ortho'` if you care about energy.

Parseval on the default normalization, accounting for the folded half:

```
sum(x**2)                                            = 5625.0
(|X[0]|^2 + 2*sum(|X[1:-1]|^2) + |X[-1]|^2) / N      = 5625.0
```

For complex input use `fft`/`ifft` with `fftfreq`, and `fftshift` to reorder the output from
`[0, positive, negative]` into `[negative, 0, positive]` for plotting. `irfft(X, n=N)` needs
`n` passed explicitly whenever `N` is odd, otherwise it assumes even and returns the wrong
length.

Related transforms: `rfft2`/`rfftn` for images and fields, `dct`/`dst` (types 1-4) for
cosine and sine transforms, `fht` for the Hankel transform, `next_fast_len(n)` to round a
length up to a fast size (`next_fast_len(2499)` returns `2500`), and `set_workers(k)` as a
context manager for threaded transforms.

## Windowing and spectral leakage

The FFT assumes the record repeats periodically. If a sinusoid does not complete a whole
number of cycles in the record, the implied discontinuity at the wrap point spreads energy
across every bin. That is leakage, and a window function tapers the record ends to suppress
it.

A 50.5 Hz tone in a 1000-point record at 1000 Hz, so exactly half a bin off center.
Normalized magnitude 20 bins away from the peak:

```
boxcar (no window)      2.16e-02
hann                    4.34e-05
blackmanharris          8.85e-07
```

Four orders of magnitude between no window and Blackman-Harris. The cost is main-lobe width:
Hann roughly doubles it, Blackman-Harris roughly quadruples it. So a window trades frequency
resolution for dynamic range. Use `hann` as a default, `blackmanharris` when a weak tone sits
near a strong one, `boxcar` only when the signal is genuinely periodic in the record or you
are measuring total energy.

`signal.get_window(name, N)` builds any of them. If you are recovering amplitudes, divide by
`win.sum()/N` (coherent gain); if you are recovering power, divide by
`sqrt(mean(win**2))` (noise gain). Getting this wrong is the other common factor-of-two.

## Power spectral density

`welch` averages the periodograms of overlapping windowed segments, trading frequency
resolution for variance reduction. Use it instead of a raw `|rfft|**2` whenever the signal is
noisy: a single periodogram has 100% standard error no matter how long the record.

```python
f, P = signal.welch(x, fs=fs, nperseg=256)
```

```
f.shape (129,)   two largest bins at 50.78 and 46.88 Hz
```

Note the second peak. The true tones are at 50 and 120 Hz, but `nperseg=256` gives
`fs/nperseg = 3.9` Hz bins, and 120 Hz is attenuated relative to the leakage skirt of the
much stronger 50 Hz tone. `nperseg` is the real knob: larger means finer frequency
resolution and noisier estimates, smaller means the reverse. `scaling='density'` (default)
returns units of `V**2/Hz`; `scaling='spectrum'` returns `V**2` per bin.

`ShortTimeFFT` is the modern class for time-frequency analysis (spectrograms, with explicit
control of the dual window and the inverse). `spectrogram` and `stft` are the older
functional forms. `csd`, `coherence`, and `periodogram` round out the family.

## Filtering

Design in second-order sections (`output='sos'`), always. The transfer-function form
(`output='ba'`) loses accuracy catastrophically above about order 8 because the polynomial
coefficients become badly conditioned.

```python
signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
```

```python
sos = signal.butter(4, 100, btype="low", fs=fs, output="sos")   # (2, 6)
y_causal = signal.sosfilt(sos, x)        # one pass, has phase lag
y_zerophase = signal.sosfiltfilt(sos, x) # forward + backward, no phase lag
```

Pass `fs=` and give `Wn` in Hz. Without `fs`, `Wn` is normalized to Nyquist, so `Wn=1.0`
means `fs/2`, and mixing the two conventions puts the cutoff in the wrong place silently.

The two filter applications are not interchangeable:

- `sosfilt` runs the difference equation once, forward. Causal, usable in real time, and it
  delays the signal by the filter's group delay (5 samples for this design, measured as the
  peak of the impulse response).
- `sosfiltfilt` runs it forward then backward. The phase cancels exactly, so features stay
  where they are, but the effective filter order doubles and it is non-causal, so it cannot
  be used online. This is what you want for offline analysis.

Filter families: `butter` (maximally flat passband), `cheby1` (ripple in passband, steeper),
`cheby2` (ripple in stopband), `ellip` (ripple in both, steepest for a given order),
`bessel` (maximally flat group delay, best for preserving waveform shape). `iirdesign` and
`buttord`/`cheb1ord` pick the order from a specification instead of you guessing.

FIR alternatives: `firwin`/`firwin2`/`remez` design, `lfilter` or `filtfilt` apply. FIR
filters have exactly linear phase and are unconditionally stable, at the cost of far higher
order for the same transition width.

Inspect what you designed with `freqz_sos(sos, fs=fs)` before trusting it (scipy 1.15+;
`sosfreqz` is the legacy name). Plain `freqz(sos, fs=fs)` treats the `(n, 6)` array as
transfer-function coefficients and raises `ValueError: Array shapes are incompatible for
broadcasting.` `group_delay` has no sos form and silently returns garbage on an sos array
(0.36 samples at 50 Hz for this design, against the correct 4.66), so call
`group_delay(sos2tf(sos), fs=fs)`.

## Convolution and correlation

`convolve` picks direct or FFT-based automatically via `choose_conv_method`. The crossover is
around a few thousand samples:

```
a: 200000 samples, b: 5000 samples, mode='full'
convolve        17.3 ms      (chose 'fft')
fftconvolve     11.8 ms
oaconvolve      11.8 ms
```

`fftconvolve` transforms both inputs at their combined length. `oaconvolve` uses overlap-add,
which wins when one input is much longer than the other and keeps memory bounded. For a
long stream against a short kernel, `oaconvolve` is the right default.

`mode=` controls the output length: `'full'` gives `len(a)+len(b)-1`, `'same'` gives
`len(a)`, `'valid'` gives only the fully-overlapping region, `len(a)-len(b)+1`. The
distinction matters more than people expect, because `'same'` centers the kernel and
therefore introduces an implicit half-kernel offset if your kernel is not symmetric.

`correlate` is convolution with one input reversed and conjugated. For cross-correlation
lags, use `correlation_lags(len(a), len(b), mode)` to get the lag axis rather than
computing offsets by hand.

## Other tools worth knowing

| task | call |
|---|---|
| local polynomial smoothing, and derivatives | `savgol_filter(x, window_length, polyorder, deriv=0)` |
| remove linear or constant trend | `detrend(x, type='linear')` |
| find peaks with prominence/width/distance criteria | `find_peaks(x, prominence=, distance=, width=)` |
| peak properties after finding | `peak_prominences`, `peak_widths` |
| resample by a rational factor | `resample_poly(x, up, down)` |
| resample to an arbitrary length | `resample(x, num)` (assumes periodicity) |
| analytic signal, envelope, instantaneous phase | `hilbert(x)` |
| median filter (impulse noise) | `medfilt`, `scipy.ndimage.median_filter` |
| N-D image filtering, morphology, labeling | `scipy.ndimage` |

`savgol_filter` deserves the callout: it fits a degree-`polyorder` polynomial by least
squares in a sliding window, so it smooths while preserving peak height and width far better
than a moving average, and `deriv=k` gives you the `k`-th derivative of the fit directly.
That is the cleanest way to differentiate noisy sampled data (see `04-floating-point.md` for
why differentiating raw noisy data is hopeless).

## Gotchas

- **`rfftfreq(N, d=1/fs)`, not `d=fs`.** The most common spectrum bug.
- **Frequency resolution is `1/T`, set by record length.** Zero-padding smooths the plot and
  adds nothing. Sampling faster raises Nyquist, not resolution.
- **Amplitude needs the `2/N` scale**, with DC and (for even `N`) Nyquist not doubled.
- **Undo the window gain.** Windowing before an amplitude measurement without dividing by the
  coherent gain under-reports amplitude by 50% for a Hann window.
- **`output='ba'` above order ~8 is numerically dead.** Use `output='sos'` and `sosfilt`
  always.
- **`Wn` without `fs=` is normalized to Nyquist.** `butter(4, 100)` with no `fs` asks for a
  cutoff 100 times above Nyquist and raises; `butter(4, 0.2)` with `fs=1000` asks for 0.2 Hz.
- **`filtfilt`/`sosfiltfilt` double the effective order** and are non-causal. A "4th-order"
  zero-phase filter has an 8th-order magnitude response.
- **`sosfilt` delays the signal.** Comparing a filtered and unfiltered trace without
  accounting for group delay makes the filter look like it shifted the physics.
- **Filtering transients at the edges.** `filtfilt`'s `padtype`/`padlen` control the edge
  extension; the defaults are usually fine, but a short signal relative to the filter length
  raises.
- **`irfft` on an odd-length original needs `n=`.** Without it the round trip silently
  returns `N-1` samples.
- **`welch` with a short `nperseg` merges nearby tones.** Check `fs/nperseg` against the
  spacing you need to resolve.
- **`mode='same'` centers the kernel.** With an asymmetric kernel that is a half-length shift
  you did not ask for.

## Quick reference

| task | call | note |
|---|---|---|
| spectrum of real data | `X = fft.rfft(x)`, `f = fft.rfftfreq(N, d=1/fs)` | `X` is `N//2+1` long |
| physical amplitude | `2*abs(X)/N`, halve bin 0 and Nyquist | |
| unitary transform | `norm='ortho'` | Parseval without extra factors |
| inverse | `fft.irfft(X, n=N)` | pass `n` for odd `N` |
| fast length | `fft.next_fast_len(n)` | |
| threaded | `with fft.set_workers(4): ...` | |
| taper before transforming | `x * signal.get_window('hann', N)` | divide by coherent gain |
| PSD of noisy data | `signal.welch(x, fs=fs, nperseg=1024)` | averages segments |
| time-frequency | `signal.ShortTimeFFT(...)` | modern spectrogram class |
| design IIR | `signal.butter(4, fc, fs=fs, output='sos')` | never `output='ba'` |
| apply, real time | `signal.sosfilt(sos, x)` | causal, has group delay |
| apply, offline | `signal.sosfiltfilt(sos, x)` | zero phase, order doubles |
| check the design | `signal.freqz_sos(sos, fs=fs)` | plot before trusting; plain `freqz` rejects sos |
| design FIR | `signal.firwin(numtaps, fc, fs=fs)` | linear phase |
| convolve, long vs short | `signal.oaconvolve(a, b)` | overlap-add |
| convolve, similar sizes | `signal.fftconvolve(a, b)` | |
| cross-correlate with lags | `signal.correlate` + `signal.correlation_lags` | |
| smooth and differentiate | `signal.savgol_filter(x, 51, 3, deriv=1, delta=dt)` | |
| peaks | `signal.find_peaks(x, prominence=..., distance=...)` | prominence beats height |
| detrend | `signal.detrend(x, type='linear')` | before spectral analysis |
