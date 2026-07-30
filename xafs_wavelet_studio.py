"""EXAFS Wavelet Studio - interactive Morlet and Cauchy analysis.

Scientific concept and interface: Dr. Esmael Balaghi
Copyright (c) 2026 Dr. Esmael Balaghi. All rights reserved.

The Morlet implementation follows H. Funke, A. C. Scheinost, and
M. Chukalina, Phys. Rev. B 71, 094110 (2005), DOI:
10.1103/PhysRevB.71.094110.

The Cauchy implementation follows M. Munoz, P. Argoul, and F. Farges,
American Mineralogist 88, 694-700 (2003), and is adapted from the
MIT-licensed xraylarch implementation by Matthew Newville et al.
https://github.com/xraypy/xraylarch/blob/master/larch/xafs/cauchy_wavelet.py

This software is a visualization and exploratory-analysis tool. Scientific
interpretation remains the user's responsibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


APP_NAME = "EXAFS Wavelet Studio"
APP_VERSION = "1.0.0"
AUTHOR = "Dr. Esmael Balaghi"
COPYRIGHT = "Copyright 2026 Dr. Esmael Balaghi"


@dataclass
class TransformSettings:
    wavelet: str = "Morlet (EXAFS)"
    kmin: float = 3.0
    kmax: float = 12.0
    rmin: float = 1.0
    rmax: float = 4.0
    k_points: int = 220
    r_points: int = 160
    eta: float = 5.0
    sigma: float = 1.0
    dk_auto: bool = True
    dk_manual: float = 0.01
    kweight: int = 0
    taper: str = "Tukey (10%)"
    cauchy_nfft: int = 2048


@dataclass
class TransformResult:
    k: np.ndarray
    r: np.ndarray
    coefficients: np.ndarray
    input_k: np.ndarray
    input_chi: np.ndarray
    processed_chi: np.ndarray
    integration_dk: float
    settings: TransformSettings
    source_name: str = ""

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.coefficients)


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Return integration weights equivalent to the composite trapezoid rule."""
    if x.size < 2:
        raise ValueError("At least two k points are required.")
    weights = np.empty_like(x, dtype=float)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    if x.size > 2:
        weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def _tukey_window(length: int, alpha: float = 0.1) -> np.ndarray:
    """Small NumPy-only Tukey window implementation."""
    if length <= 1:
        return np.ones(max(length, 1), dtype=float)
    if alpha <= 0:
        return np.ones(length, dtype=float)
    if alpha >= 1:
        return np.hanning(length)
    x = np.linspace(0.0, 1.0, length)
    window = np.ones(length, dtype=float)
    left = x < alpha / 2.0
    right = x >= 1.0 - alpha / 2.0
    window[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[left] - alpha / 2.0)))
    window[right] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[right] - 1.0 + alpha / 2.0)))
    return window


def _clean_xy(k: np.ndarray, chi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float).ravel()
    chi = np.asarray(chi, dtype=float).ravel()
    if k.size != chi.size:
        raise ValueError("The k and chi columns have different lengths.")
    finite = np.isfinite(k) & np.isfinite(chi)
    k, chi = k[finite], chi[finite]
    if k.size < 4:
        raise ValueError("The file must contain at least four finite k and chi rows.")
    order = np.argsort(k)
    k, chi = k[order], chi[order]
    unique_k, inverse, counts = np.unique(k, return_inverse=True, return_counts=True)
    if unique_k.size != k.size:
        sums = np.bincount(inverse, weights=chi)
        chi = sums / counts
        k = unique_k
    if np.any(np.diff(k) <= 0):
        raise ValueError("k values could not be converted to a strictly increasing array.")
    return k, chi


def prepare_signal(
    k: np.ndarray, chi: np.ndarray, settings: TransformSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """Validate, crop, weight, and taper input data.

    Returns selected k, raw chi, processed chi, representative dk, and either
    trapezoid weights (automatic mode) or None (manual rectangular spacing).
    """
    k, chi = _clean_xy(k, chi)
    if settings.kmin >= settings.kmax:
        raise ValueError("k max must be larger than k min.")
    if settings.rmin <= 0 or settings.rmax <= settings.rmin:
        raise ValueError("R min must be positive and R max must be larger than R min.")
    if settings.kmin < k[0] or settings.kmax > k[-1]:
        raise ValueError(
            f"The requested k range [{settings.kmin:g}, {settings.kmax:g}] is outside "
            f"the data range [{k[0]:g}, {k[-1]:g}] A^-1."
        )
    mask = (k >= settings.kmin) & (k <= settings.kmax)
    selected_k, raw_chi = k[mask], chi[mask]
    if selected_k.size < 8:
        raise ValueError("The selected k range contains fewer than eight data points.")
    spacings = np.diff(selected_k)
    representative_dk = float(np.median(spacings))
    if representative_dk <= 0:
        raise ValueError("The k spacing must be positive.")
    processed = raw_chi * np.power(selected_k, settings.kweight)
    if settings.taper == "Hann":
        processed = processed * np.hanning(selected_k.size)
    elif settings.taper == "Tukey (10%)":
        processed = processed * _tukey_window(selected_k.size, 0.1)
    elif settings.taper != "None":
        raise ValueError(f"Unknown taper: {settings.taper}")
    if settings.dk_auto:
        weights = _trapezoid_weights(selected_k)
    else:
        if settings.dk_manual <= 0:
            raise ValueError("Manual dk must be larger than zero.")
        representative_dk = float(settings.dk_manual)
        weights = None
    return selected_k, raw_chi, processed, representative_dk, weights


def morlet_exafs_transform(
    k: np.ndarray,
    signal: np.ndarray,
    settings: TransformSettings,
    weights: Optional[np.ndarray],
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous Morlet transform using the EXAFS k-to-R convention."""
    if settings.eta <= 0 or settings.sigma <= 0:
        raise ValueError("Morlet eta and sigma must both be larger than zero.")
    if settings.k_points < 40 or settings.r_points < 30:
        raise ValueError("Use at least 40 k pixels and 30 R pixels for a useful map.")
    b_values = np.linspace(settings.kmin, settings.kmax, settings.k_points)
    r_values = np.linspace(settings.rmin, settings.rmax, settings.r_points)
    scales = settings.eta / (2.0 * r_values)
    integration_weights = (
        np.full(k.size, settings.dk_manual, dtype=float) if weights is None else weights
    )
    weighted_signal = signal * integration_weights
    correction = math.exp(-0.5 * (settings.eta * settings.sigma) ** 2)
    normalization = 1.0 / (math.sqrt(2.0 * math.pi) * settings.sigma)
    output = np.empty((r_values.size, b_values.size), dtype=np.complex128)
    chunk_size = 12
    for start in range(0, r_values.size, chunk_size):
        stop = min(start + chunk_size, r_values.size)
        a = scales[start:stop, None, None]
        u = (k[None, None, :] - b_values[None, :, None]) / a
        wavelet = normalization * (
            np.exp(1j * settings.eta * u) - correction
        ) * np.exp(-0.5 * np.square(u / settings.sigma))
        output[start:stop] = np.sum(
            wavelet * weighted_signal[None, None, :] / np.sqrt(a), axis=2
        )
        if progress:
            progress(stop / r_values.size, "Calculating Morlet map")
    return b_values, r_values, output


def _interp_complex_rows(
    source_x: np.ndarray, values: np.ndarray, target_x: np.ndarray
) -> np.ndarray:
    out = np.empty((values.shape[0], target_x.size), dtype=np.complex128)
    for row in range(values.shape[0]):
        out[row] = np.interp(target_x, source_x, values[row].real) + 1j * np.interp(
            target_x, source_x, values[row].imag
        )
    return out


def cauchy_exafs_transform(
    k: np.ndarray,
    signal: np.ndarray,
    settings: TransformSettings,
    dk: float,
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous Cauchy transform adapted from xraylarch's EXAFS routine.

    The source routine expects a uniform k grid beginning at zero. This version
    resamples the selected signal and zero-pads the unmeasured low-k region.
    """
    requested_nfft = int(settings.cauchy_nfft)
    if requested_nfft < 512:
        raise ValueError("Cauchy FFT size must be at least 512.")
    if dk <= 0:
        raise ValueError("dk must be positive for the Cauchy transform.")
    full_k = np.arange(0.0, settings.kmax + 0.5 * dk, dk)
    full_signal = np.zeros(full_k.size, dtype=float)
    measured = (full_k >= k[0]) & (full_k <= k[-1])
    full_signal[measured] = np.interp(full_k[measured], k, signal)
    minimum_nfft = 2 ** math.ceil(math.log2(max(512, 2 * full_k.size)))
    nfft = max(requested_nfft, minimum_nfft)
    half = nfft // 2
    xnew = np.zeros(half, dtype=float)
    usable = min(full_signal.size, half)
    xnew[:usable] = full_signal[:usable]
    rstep = (math.pi / nfft) / dk
    nrpts = max(32, int(round(settings.rmax / rstep)))
    r_full = np.linspace(0.0, settings.rmax, nrpts)
    safe_r = r_full.copy()
    safe_r[0] = 1.0e-19
    scale = nrpts / (2.0 * safe_r)
    frequency = (1.0 / dk) * np.arange(nfft) / (2.0 * nfft)
    omega = 2.0 * np.pi * frequency
    transform_fft = np.fft.fft(xnew, n=2 * nfft)
    log_norm = math.log(2.0 * math.pi) - math.lgamma(nrpts + 1.0)
    output = np.empty((nrpts, usable), dtype=np.complex128)
    tiny = np.finfo(float).tiny
    for index in range(nrpts):
        scaled_omega = np.maximum(scale[index] * omega, tiny)
        log_filter = log_norm + nrpts * np.log(scaled_omega) - scaled_omega
        spectral_filter = np.exp(np.clip(log_filter, -745.0, 50.0))
        output[index] = np.fft.ifft(
            np.conjugate(spectral_filter) * transform_fft[:nfft], n=2 * nfft
        )[:usable]
        if progress and (index % max(1, nrpts // 25) == 0 or index == nrpts - 1):
            progress((index + 1) / nrpts, "Calculating Cauchy map")
    map_k = full_k[:usable]
    display_k = np.linspace(settings.kmin, settings.kmax, settings.k_points)
    r_mask = r_full >= settings.rmin
    if np.count_nonzero(r_mask) < 2:
        raise ValueError("The selected R range has too few Cauchy samples; increase FFT size.")
    cropped_r = r_full[r_mask]
    cropped_output = output[r_mask]
    k_resampled = _interp_complex_rows(map_k, cropped_output, display_k)
    display_r = np.linspace(settings.rmin, settings.rmax, settings.r_points)
    final = np.empty((display_r.size, display_k.size), dtype=np.complex128)
    for col in range(display_k.size):
        final[:, col] = np.interp(display_r, cropped_r, k_resampled[:, col].real) + 1j * np.interp(
            display_r, cropped_r, k_resampled[:, col].imag
        )
    return display_k, display_r, final


def calculate_transform(
    k: np.ndarray,
    chi: np.ndarray,
    settings: TransformSettings,
    source_name: str = "",
    progress: Optional[Callable[[float, str], None]] = None,
) -> TransformResult:
    selected_k, raw_chi, processed, representative_dk, weights = prepare_signal(
        k, chi, settings
    )
    if settings.wavelet == "Morlet (EXAFS)":
        out_k, out_r, coefficients = morlet_exafs_transform(
            selected_k, processed, settings, weights, progress
        )
    elif settings.wavelet == "Cauchy (EXAFS)":
        out_k, out_r, coefficients = cauchy_exafs_transform(
            selected_k, processed, settings, representative_dk, progress
        )
    else:
        raise ValueError(f"Unknown transform: {settings.wavelet}")
    return TransformResult(
        k=out_k,
        r=out_r,
        coefficients=coefficients,
        input_k=selected_k,
        input_chi=raw_chi,
        processed_chi=processed,
        integration_dk=representative_dk,
        settings=settings,
        source_name=source_name,
    )


def generate_demo_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic EXAFS-like teaching signal."""
    k = np.arange(2.0, 14.0001, 0.04)
    envelope = np.exp(-0.035 * np.square(k - 2.0))
    shell_1 = 0.68 * np.sin(2.0 * k * 2.05 + 0.30)
    shell_2 = 0.38 * np.sin(2.0 * k * 3.10 - 0.85)
    slow = 0.12 * np.sin(2.0 * k * 1.25 + 1.40)
    rng = np.random.default_rng(2026)
    chi = envelope * (shell_1 + shell_2 + slow) + rng.normal(0.0, 0.012, k.size)
    return k, chi


def load_numeric_file(
    path: str, delimiter_name: str = "Auto / whitespace", skip_rows: int = 0,
    k_column: int = 1, chi_column: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Load two numeric columns from common Athena/text/CSV exports."""
    if k_column < 1 or chi_column < 1:
        raise ValueError("Column numbers start at 1.")
    delimiter = {
        "Auto / whitespace": None,
        "Comma": ",",
        "Semicolon": ";",
        "Tab": "\t",
    }.get(delimiter_name)
    if delimiter_name not in {"Auto / whitespace", "Comma", "Semicolon", "Tab"}:
        raise ValueError(f"Unknown delimiter option: {delimiter_name}")
    data = np.genfromtxt(
        path,
        comments="#",
        delimiter=delimiter,
        skip_header=max(0, int(skip_rows)),
        invalid_raise=False,
        autostrip=True,
    )
    if data.ndim == 1:
        if data.size == 0:
            raise ValueError("No numeric data were found in the selected file.")
        data = data.reshape(1, -1)
    highest = max(k_column, chi_column)
    if data.shape[1] < highest:
        raise ValueError(
            f"The file has {data.shape[1]} readable columns, but column {highest} was requested."
        )
    return _clean_xy(data[:, k_column - 1], data[:, chi_column - 1])


def _component_matrix(result: TransformResult, component: str) -> np.ndarray:
    if component == "Magnitude":
        return np.abs(result.coefficients)
    if component == "Power":
        return np.square(np.abs(result.coefficients))
    if component == "Real":
        return result.coefficients.real
    if component == "Imaginary":
        return result.coefficients.imag
    if component == "Phase":
        return np.angle(result.coefficients)
    raise ValueError(f"Unknown plot component: {component}")


def export_transform(path: str, result: TransformResult) -> None:
    """Export result as compressed NPZ or long-form delimited text."""
    suffix = Path(path).suffix.lower()
    metadata = asdict(result.settings)
    metadata.update(
        source=result.source_name,
        effective_dk=result.integration_dk,
        app=APP_NAME,
        version=APP_VERSION,
        author=AUTHOR,
    )
    if suffix == ".npz":
        np.savez_compressed(
            path,
            k=result.k,
            r=result.r,
            coefficients=result.coefficients,
            magnitude=result.magnitude,
            input_k=result.input_k,
            input_chi=result.input_chi,
            processed_chi=result.processed_chi,
            metadata=json.dumps(metadata),
        )
        return
    delimiter = "," if suffix == ".csv" else "\t"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["k_A^-1", "R_A", "magnitude", "real", "imaginary", "phase_rad"])
        for ri, rval in enumerate(result.r):
            row = result.coefficients[ri]
            for ki, kval in enumerate(result.k):
                value = row[ki]
                writer.writerow(
                    [
                        f"{kval:.9g}",
                        f"{rval:.9g}",
                        f"{abs(value):.9g}",
                        f"{value.real:.9g}",
                        f"{value.imag:.9g}",
                        f"{np.angle(value):.9g}",
                    ]
                )


def render_analysis_figure(
    figure,
    result: TransformResult,
    component: str = "Magnitude",
    colormap: str = "turbo",
    levels: int = 80,
    log_scale: bool = False,
    normalize: bool = False,
    show_contours: bool = False,
):
    """Render the complete analysis view and return the map axes and displayed data."""
    figure.clear()
    figure.set_facecolor("#f8fafc")
    grid = figure.add_gridspec(
        2, 2, height_ratios=(1.0, 3.0), width_ratios=(5.0, 1.15),
        left=0.075, right=0.955, bottom=0.085, top=0.93, hspace=0.30, wspace=0.25,
    )
    signal_ax = figure.add_subplot(grid[0, :])
    map_ax = figure.add_subplot(grid[1, 0])
    projection_ax = figure.add_subplot(grid[1, 1], sharey=map_ax)
    for axis in (signal_ax, map_ax, projection_ax):
        axis.set_facecolor("#ffffff")
        axis.tick_params(colors="#334155", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#cbd5e1")
    signal_ax.plot(
        result.input_k, result.processed_chi, color="#0f766e", linewidth=1.35,
        label=f"Processed k^{result.settings.kweight} chi(k)",
    )
    signal_ax.axhline(0.0, color="#cbd5e1", linewidth=0.8)
    signal_ax.set_xlim(result.k[0], result.k[-1])
    signal_ax.set_ylabel("Signal", color="#334155", fontsize=9)
    signal_ax.set_title(
        f"{result.settings.wavelet}  |  {result.source_name or 'demonstration data'}",
        loc="left", color="#0f172a", fontsize=11, fontweight="bold", pad=8,
    )
    signal_ax.grid(color="#e2e8f0", linewidth=0.6, alpha=0.65)
    magnitude = np.abs(result.coefficients)
    k_projection = np.trapz(magnitude, result.r, axis=0)
    if np.max(k_projection) > 0:
        k_projection = k_projection / np.max(k_projection)
    twin = signal_ax.twinx()
    twin.plot(result.k, k_projection, color="#d97706", linewidth=1.0, alpha=0.8)
    twin.set_ylabel("WT k projection", color="#b45309", fontsize=8)
    twin.tick_params(colors="#b45309", labelsize=7)
    twin.set_ylim(0.0, 1.08)

    display = _component_matrix(result, component).astype(float, copy=True)
    if normalize and component != "Phase":
        denominator = float(np.nanmax(np.abs(display)))
        if denominator > 0:
            display /= denominator
    colorbar_label = component
    if log_scale:
        if component not in {"Magnitude", "Power"}:
            raise ValueError("Log scale is available only for Magnitude or Power.")
        maximum = float(np.nanmax(display))
        floor = max(maximum * 1.0e-6, np.finfo(float).tiny)
        display = np.log10(np.maximum(display, floor))
        colorbar_label = f"log10({component.lower()})"
    contour = map_ax.contourf(
        result.k, result.r, display, levels=max(8, int(levels)), cmap=colormap,
    )
    if show_contours:
        line_levels = max(4, min(14, int(levels) // 8))
        lines = map_ax.contour(
            result.k, result.r, display, levels=line_levels,
            colors="#0f172a", linewidths=0.45, alpha=0.48,
        )
        map_ax.clabel(lines, inline=True, fontsize=6, fmt="%.2g")
    colorbar = figure.colorbar(contour, ax=map_ax, pad=0.018, fraction=0.046)
    colorbar.set_label(colorbar_label + (" (normalized)" if normalize else ""), fontsize=8)
    colorbar.ax.tick_params(labelsize=7, colors="#334155")
    map_ax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)", fontsize=10, color="#0f172a")
    map_ax.set_ylabel(r"$R$ ($\mathrm{\AA}$)", fontsize=10, color="#0f172a")
    map_ax.set_xlim(result.k[0], result.k[-1])
    map_ax.set_ylim(result.r[0], result.r[-1])
    map_ax.set_title(f"Wavelet {component.lower()} map", loc="left", fontsize=10, color="#0f172a")

    r_projection = np.trapz(magnitude, result.k, axis=1)
    if np.max(r_projection) > 0:
        r_projection = r_projection / np.max(r_projection)
    projection_ax.plot(r_projection, result.r, color="#7c3aed", linewidth=1.4)
    projection_ax.fill_betweenx(result.r, 0.0, r_projection, color="#8b5cf6", alpha=0.14)
    projection_ax.set_xlabel("R proj.", fontsize=8, color="#334155")
    projection_ax.set_xlim(0.0, 1.08)
    projection_ax.grid(color="#e2e8f0", linewidth=0.6, alpha=0.65)
    projection_ax.tick_params(labelleft=False)
    figure.text(
        0.955, 0.018, f"{COPYRIGHT}  |  v{APP_VERSION}",
        ha="right", va="bottom", fontsize=6.5, color="#64748b",
    )
    return map_ax, display


def _self_test() -> int:
    k, chi = generate_demo_data()
    morlet = TransformSettings(kmin=3.0, kmax=12.0, rmin=0.8, rmax=4.5, k_points=70, r_points=55)
    result_m = calculate_transform(k, chi, morlet, "self-test")
    assert result_m.coefficients.shape == (55, 70)
    assert np.isfinite(result_m.coefficients).all()
    peak_r = result_m.r[np.argmax(np.sum(result_m.magnitude, axis=1))]
    assert 1.5 < peak_r < 3.6, peak_r
    cauchy = TransformSettings(
        wavelet="Cauchy (EXAFS)", kmin=3.0, kmax=12.0, rmin=0.8,
        rmax=4.5, k_points=64, r_points=50, cauchy_nfft=1024,
    )
    result_c = calculate_transform(k, chi, cauchy, "self-test")
    assert result_c.coefficients.shape == (50, 64)
    assert np.isfinite(result_c.coefficients).all()
    assert float(np.max(result_c.magnitude)) > 0.0
    print(
        f"Self-test passed: Morlet {result_m.coefficients.shape}, "
        f"Cauchy {result_c.coefficients.shape}, Morlet peak R={peak_r:.3f} A"
    )
    return 0


def _launch_gui() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    class Tooltip:
        def __init__(self, widget, text: str):
            self.widget = widget
            self.text = text
            self.window = None
            widget.bind("<Enter>", self.show, add="+")
            widget.bind("<Leave>", self.hide, add="+")

        def show(self, _event=None):
            if self.window or not self.text:
                return
            x = self.widget.winfo_rootx() + self.widget.winfo_width() + 8
            y = self.widget.winfo_rooty() - 4
            self.window = tk.Toplevel(self.widget)
            self.window.wm_overrideredirect(True)
            self.window.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                self.window, text=self.text, justify="left", wraplength=390,
                background="#fefce8", foreground="#1e293b", relief="solid",
                borderwidth=1, padx=10, pady=8, font=("Segoe UI", 9),
            )
            label.pack()

        def hide(self, _event=None):
            if self.window:
                self.window.destroy()
                self.window = None

    class WaveletStudio(tk.Tk):
        BG = "#0b1220"
        PANEL = "#111827"
        PANEL_2 = "#162033"
        BORDER = "#273449"
        TEXT = "#e5e7eb"
        MUTED = "#94a3b8"
        ACCENT = "#14b8a6"
        ACCENT_DARK = "#0f766e"

        def __init__(self):
            super().__init__()
            self.title(f"{APP_NAME} {APP_VERSION}")
            self._set_window_icon(tk)
            self.geometry("1440x900")
            self.minsize(1120, 720)
            self.configure(background=self.BG)
            self.protocol("WM_DELETE_WINDOW", self._close)
            self.result: Optional[TransformResult] = None
            self.loaded_k: Optional[np.ndarray] = None
            self.loaded_chi: Optional[np.ndarray] = None
            self.source_name = ""
            self._worker: Optional[threading.Thread] = None
            self._map_axis = None
            self._display_matrix = None
            self._configure_styles(ttk)
            self._variables(tk)
            self._build_ui(tk, ttk, filedialog, messagebox, Figure, FigureCanvasTkAgg, NavigationToolbar2Tk)
            self.after(150, self.load_demo)

        def _set_window_icon(self, tk_module):
            """Set the same icon in source and one-file PyInstaller builds."""
            bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
            icon_path = bundle_root / "EXAFS_Wavelet_Studio.png"
            try:
                self._icon_image = tk_module.PhotoImage(file=str(icon_path))
                self.iconphoto(True, self._icon_image)
            except (tk_module.TclError, OSError):
                self._icon_image = None

        def _configure_styles(self, ttk_module):
            style = ttk_module.Style(self)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            style.configure("TEntry", fieldbackground="#f8fafc", foreground="#0f172a", padding=5)
            style.configure("TCombobox", fieldbackground="#f8fafc", foreground="#0f172a", padding=4)
            style.configure("Accent.TButton", background=self.ACCENT, foreground="#062c2a", padding=(14, 8), font=("Segoe UI Semibold", 9))
            style.map("Accent.TButton", background=[("active", "#2dd4bf"), ("disabled", "#475569")])
            style.configure("Quiet.TButton", background="#243147", foreground=self.TEXT, padding=(10, 7))
            style.map("Quiet.TButton", background=[("active", "#334155")])
            style.configure("Dark.TCheckbutton", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 9))
            style.map("Dark.TCheckbutton", background=[("active", self.PANEL)], foreground=[("disabled", "#64748b")])
            style.configure("Horizontal.TProgressbar", troughcolor="#263247", background=self.ACCENT)

        def _variables(self, tk_module):
            self.vars = {
                "file": tk_module.StringVar(),
                "delimiter": tk_module.StringVar(value="Auto / whitespace"),
                "skip_rows": tk_module.StringVar(value="0"),
                "k_column": tk_module.StringVar(value="1"),
                "chi_column": tk_module.StringVar(value="2"),
                "kweight": tk_module.StringVar(value="0"),
                "taper": tk_module.StringVar(value="Tukey (10%)"),
                "wavelet": tk_module.StringVar(value="Morlet (EXAFS)"),
                "kmin": tk_module.StringVar(value="3.0"),
                "kmax": tk_module.StringVar(value="12.0"),
                "rmin": tk_module.StringVar(value="1.0"),
                "rmax": tk_module.StringVar(value="4.0"),
                "k_points": tk_module.StringVar(value="220"),
                "r_points": tk_module.StringVar(value="160"),
                "dk_auto": tk_module.BooleanVar(value=True),
                "dk_manual": tk_module.StringVar(value="0.01"),
                "eta": tk_module.StringVar(value="5.0"),
                "sigma": tk_module.StringVar(value="1.0"),
                "cauchy_nfft": tk_module.StringVar(value="2048"),
                "component": tk_module.StringVar(value="Magnitude"),
                "colormap": tk_module.StringVar(value="turbo"),
                "levels": tk_module.StringVar(value="80"),
                "log_scale": tk_module.BooleanVar(value=False),
                "normalize": tk_module.BooleanVar(value=False),
                "contours": tk_module.BooleanVar(value=False),
            }
            self.status = tk_module.StringVar(value="Load a file or start with the built-in demonstration.")
            self.cursor_status = tk_module.StringVar(value="")
            self.progress_value = tk_module.DoubleVar(value=0.0)

        def _build_ui(self, tk_module, ttk_module, filedialog_module, messagebox_module, FigureClass, CanvasClass, ToolbarClass):
            self.filedialog = filedialog_module
            self.messagebox = messagebox_module
            header = tk_module.Frame(self, bg=self.BG, height=72)
            header.pack(fill="x")
            header.pack_propagate(False)
            title_box = tk_module.Frame(header, bg=self.BG)
            title_box.pack(side="left", padx=24, pady=13)
            tk_module.Label(title_box, text="EXAFS", bg=self.BG, fg=self.ACCENT, font=("Segoe UI Semibold", 10)).pack(anchor="w")
            tk_module.Label(title_box, text="Wavelet Studio", bg=self.BG, fg="#f8fafc", font=("Segoe UI Semibold", 20)).pack(anchor="w")
            badge = tk_module.Frame(header, bg="#142c35", highlightbackground="#1c5b5a", highlightthickness=1)
            badge.pack(side="right", padx=24, pady=14)
            tk_module.Label(badge, text="SCIENTIFIC CONCEPT & INTERFACE", bg="#142c35", fg="#67e8f9", font=("Segoe UI", 7)).pack(padx=13, pady=(7, 0))
            tk_module.Label(badge, text=AUTHOR, bg="#142c35", fg="#f0fdfa", font=("Segoe UI Semibold", 10)).pack(padx=13, pady=(0, 7))

            body = tk_module.PanedWindow(self, orient="horizontal", sashwidth=5, bg=self.BG, bd=0)
            body.pack(fill="both", expand=True, padx=(16, 16), pady=(0, 10))
            side_holder = tk_module.Frame(body, bg=self.PANEL, width=360)
            side_holder.pack_propagate(False)
            body.add(side_holder, minsize=320, width=360)
            plot_holder = tk_module.Frame(body, bg="#e2e8f0")
            body.add(plot_holder, minsize=700)

            sidebar_canvas = tk_module.Canvas(side_holder, bg=self.PANEL, highlightthickness=0, bd=0)
            scrollbar = ttk_module.Scrollbar(side_holder, orient="vertical", command=sidebar_canvas.yview)
            sidebar_canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            sidebar_canvas.pack(side="left", fill="both", expand=True)
            self.sidebar = tk_module.Frame(sidebar_canvas, bg=self.PANEL)
            self.sidebar_window = sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
            self.sidebar.bind("<Configure>", lambda _e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
            sidebar_canvas.bind("<Configure>", lambda e: sidebar_canvas.itemconfigure(self.sidebar_window, width=e.width))
            sidebar_canvas.bind("<Enter>", lambda _e: sidebar_canvas.bind_all("<MouseWheel>", lambda ev: sidebar_canvas.yview_scroll(int(-ev.delta / 120), "units")))
            sidebar_canvas.bind("<Leave>", lambda _e: sidebar_canvas.unbind_all("<MouseWheel>"))

            self._build_data_section(tk_module, ttk_module)
            self._build_transform_section(tk_module, ttk_module)
            self._build_plot_section(tk_module, ttk_module)
            action = tk_module.Frame(self.sidebar, bg=self.PANEL)
            action.pack(fill="x", padx=14, pady=(4, 12))
            self.run_button = ttk_module.Button(action, text="RUN WAVELET ANALYSIS", style="Accent.TButton", command=self.run_analysis)
            self.run_button.pack(fill="x")
            self.progress = ttk_module.Progressbar(action, variable=self.progress_value, maximum=100.0)
            self.progress.pack(fill="x", pady=(9, 0))
            tk_module.Label(action, textvariable=self.status, bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 8), wraplength=315, justify="left").pack(fill="x", pady=(7, 0))

            topbar = tk_module.Frame(plot_holder, bg="#eef2f7", height=48)
            topbar.pack(fill="x")
            topbar.pack_propagate(False)
            ttk_module.Button(topbar, text="Export figure", style="Quiet.TButton", command=self.export_figure).pack(side="left", padx=(12, 5), pady=8)
            ttk_module.Button(topbar, text="Export data", style="Quiet.TButton", command=self.export_data).pack(side="left", padx=5, pady=8)
            ttk_module.Button(topbar, text="Quick guide", style="Quiet.TButton", command=self.show_guide).pack(side="right", padx=5, pady=8)
            ttk_module.Button(topbar, text="About", style="Quiet.TButton", command=self.show_about).pack(side="right", padx=(5, 12), pady=8)
            tk_module.Label(topbar, textvariable=self.cursor_status, bg="#eef2f7", fg="#475569", font=("Consolas", 8)).pack(side="right", padx=12)

            self.figure = Figure(figsize=(10, 7), dpi=100, facecolor="#f8fafc")
            self.canvas = CanvasClass(self.figure, master=plot_holder)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar_frame = tk_module.Frame(plot_holder, bg="#e2e8f0")
            toolbar_frame.pack(fill="x")
            self.toolbar = ToolbarClass(self.canvas, toolbar_frame, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side="left", padx=6)
            self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
            self._draw_welcome()

        def _section(self, tk_module, title: str, subtitle: str):
            outer = tk_module.Frame(self.sidebar, bg=self.PANEL_2, highlightbackground=self.BORDER, highlightthickness=1)
            outer.pack(fill="x", padx=12, pady=(10, 0))
            tk_module.Label(outer, text=title, bg=self.PANEL_2, fg="#f8fafc", font=("Segoe UI Semibold", 10)).pack(anchor="w", padx=12, pady=(10, 0))
            tk_module.Label(outer, text=subtitle, bg=self.PANEL_2, fg=self.MUTED, font=("Segoe UI", 8), wraplength=305, justify="left").pack(anchor="w", padx=12, pady=(1, 8))
            content = tk_module.Frame(outer, bg=self.PANEL_2)
            content.pack(fill="x", padx=12, pady=(0, 10))
            content.grid_columnconfigure(1, weight=1)
            return content

        def _help(self, tk_module, parent, row: int, text: str):
            mark = tk_module.Label(parent, text="?", width=2, bg="#203047", fg="#67e8f9", cursor="hand2", font=("Segoe UI Semibold", 8))
            mark.grid(row=row, column=2, padx=(6, 0), pady=4, sticky="n")
            Tooltip(mark, text)
            mark.bind("<Button-1>", lambda _e: self.messagebox.showinfo("Parameter help", text), add="+")

        def _field(self, tk_module, ttk_module, parent, row: int, label: str, key: str, help_text: str, values=None, width=12, command=None):
            tk_module.Label(parent, text=label, bg=self.PANEL_2, fg="#cbd5e1", font=("Segoe UI", 8)).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            if values is None:
                widget = ttk_module.Entry(parent, textvariable=self.vars[key], width=width)
            else:
                widget = ttk_module.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly", width=width)
                if command:
                    widget.bind("<<ComboboxSelected>>", lambda _e: command())
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            self._help(tk_module, parent, row, help_text)
            return widget

        def _build_data_section(self, tk_module, ttk_module):
            section = self._section(tk_module, "1  DATA", "Load k and chi(k), then define minimal preprocessing.")
            tk_module.Label(section, text="Input file", bg=self.PANEL_2, fg="#cbd5e1", font=("Segoe UI", 8)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
            file_row = tk_module.Frame(section, bg=self.PANEL_2)
            file_row.grid(row=0, column=1, sticky="ew", pady=4)
            ttk_module.Entry(file_row, textvariable=self.vars["file"]).pack(side="left", fill="x", expand=True)
            ttk_module.Button(file_row, text="...", width=3, command=self.browse_file).pack(side="left", padx=(4, 0))
            self._help(tk_module, section, 0, "A text file containing at least two numeric columns. The usual choice is k in A^-1 and chi(k), or an already k-weighted chi(k), in the next column. Lines beginning with # are ignored.")
            self._field(tk_module, ttk_module, section, 1, "Delimiter", "delimiter", "How columns are separated. 'Auto / whitespace' handles spaces and tabs and is right for most Athena exports. Choose comma or semicolon for CSV-style files.", ["Auto / whitespace", "Comma", "Semicolon", "Tab"], 16)
            self._field(tk_module, ttk_module, section, 2, "Skip rows", "skip_rows", "Number of physical lines to ignore at the top before reading. Leave 0 when headers already begin with #.")
            self._field(tk_module, ttk_module, section, 3, "k column", "k_column", "One-based column number containing photoelectron wavenumber k in inverse angstroms. The first column is 1.")
            self._field(tk_module, ttk_module, section, 4, "chi column", "chi_column", "One-based column number containing chi(k) or a pre-weighted EXAFS signal. The second column is 2.")
            self._field(tk_module, ttk_module, section, 5, "k weight", "kweight", "Multiply the loaded signal by k^n before transforming. Use 0 if the selected column is already weighted (for example k^2 chi). Use 1, 2, or 3 for unweighted chi(k), depending on your analysis convention.", ["0", "1", "2", "3"], 10)
            self._field(tk_module, ttk_module, section, 6, "Edge taper", "taper", "Reduces artificial intensity at the k-range boundaries. Tukey 10% preserves most of the data while gently turning down both ends; Hann is stronger; None exactly preserves the selected samples.", ["Tukey (10%)", "Hann", "None"], 16)
            demo = ttk_module.Button(section, text="Use demonstration signal", style="Quiet.TButton", command=self.load_demo)
            demo.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        def _build_transform_section(self, tk_module, ttk_module):
            section = self._section(tk_module, "2  TRANSFORM", "Choose the EXAFS method and the k-R analysis window.")
            self._field(tk_module, ttk_module, section, 0, "Wavelet", "wavelet", "Morlet is the Funke-Scheinost-Chukalina EXAFS formulation and exposes eta and sigma. Cauchy follows the Munoz-Argoul-Farges transform used by Larch and exposes FFT resolution.", ["Morlet (EXAFS)", "Cauchy (EXAFS)"], 18, self._wavelet_changed)
            self._field(tk_module, ttk_module, section, 1, "k min (A^-1)", "kmin", "Lowest measured k included in the analysis. It must lie inside the loaded data range. A common EXAFS starting value is near 3 A^-1, but inspect your data quality.")
            self._field(tk_module, ttk_module, section, 2, "k max (A^-1)", "kmax", "Highest measured k included in the analysis. Do not extend it into a region dominated by noise.")
            self._field(tk_module, ttk_module, section, 3, "R min (A)", "rmin", "Lowest apparent radial-distance coordinate shown in the wavelet map. This is not automatically phase-shift corrected, so do not read it as an exact bond length.")
            self._field(tk_module, ttk_module, section, 4, "R max (A)", "rmax", "Highest apparent radial-distance coordinate shown in the map. Choose a focused range for clearer contrast and faster calculation.")
            self._field(tk_module, ttk_module, section, 5, "k pixels", "k_points", "Horizontal resolution of the calculated display grid. 180-300 is usually smooth enough. Larger values increase calculation time and export size.")
            self._field(tk_module, ttk_module, section, 6, "R pixels", "r_points", "Vertical resolution of the calculated display grid. 120-220 is usually smooth enough. This changes sampling, not the physical resolution of the wavelet.")
            auto_row = tk_module.Frame(section, bg=self.PANEL_2)
            auto_row.grid(row=7, column=0, columnspan=2, sticky="ew", pady=3)
            ttk_module.Checkbutton(auto_row, text="Auto dk from data", variable=self.vars["dk_auto"], style="Dark.TCheckbutton", command=self._dk_changed).pack(side="left")
            self._help(tk_module, section, 7, "dk is the numerical integration step in k, not a universal constant. Auto mode is recommended: Morlet uses the actual interval of every data pair (trapezoid integration), while Cauchy resamples to the median spacing because its FFT requires a uniform grid. Turn Auto off only to reproduce a known fixed-step calculation.")
            self.dk_entry = self._field(tk_module, ttk_module, section, 8, "Manual dk", "dk_manual", "Fixed k increment in A^-1 used only when Auto dk is off. The legacy script used 0.01. A wrong value rescales Morlet amplitudes; for Cauchy it also changes resampling and the R grid.")
            self.dynamic = tk_module.Frame(section, bg=self.PANEL_2)
            self.dynamic.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(6, 0))
            self.dynamic.grid_columnconfigure(1, weight=1)
            self._wavelet_changed()
            self._dk_changed()

        def _build_plot_section(self, tk_module, ttk_module):
            section = self._section(tk_module, "3  PLOT", "Change the view without recalculating the transform.")
            self._field(tk_module, ttk_module, section, 0, "Component", "component", "Magnitude is the usual qualitative EXAFS wavelet map. Power emphasizes strong features. Real and Imaginary retain phase-sensitive signed information. Phase shows angle in radians.", ["Magnitude", "Power", "Real", "Imaginary", "Phase"], 14, self.refresh_plot)
            self._field(tk_module, ttk_module, section, 1, "Color map", "colormap", "Color palette for the 2D map. Turbo and Viridis are perceptually ordered; Magma is dark-to-light; RdBu_r is useful for signed real or imaginary values.", ["turbo", "viridis", "magma", "cividis", "RdBu_r", "Spectral_r"], 14, self.refresh_plot)
            self._field(tk_module, ttk_module, section, 2, "Color levels", "levels", "Number of filled contour levels. 60-120 looks smooth; lower values reveal contour bands and render faster.")
            options = [
                ("Log color scale", "log_scale", "Shows log10 magnitude or power so weak features remain visible next to strong peaks. It is intentionally unavailable for signed real/imaginary values and phase."),
                ("Normalize to max", "normalize", "Divide the displayed component by its largest absolute value. This changes only the plot, not exported complex coefficients."),
                ("Contour overlay", "contours", "Draw labeled isolines on top of the filled color map. Useful for printouts, but visually busy for noisy spectra."),
            ]
            for row, (label, key, help_text) in enumerate(options, start=3):
                ttk_module.Checkbutton(section, text=label, variable=self.vars[key], style="Dark.TCheckbutton", command=self.refresh_plot).grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
                self._help(tk_module, section, row, help_text)
            ttk_module.Button(section, text="Apply plot settings", style="Quiet.TButton", command=self.refresh_plot).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(7, 0))

        def _wavelet_changed(self):
            for child in self.dynamic.winfo_children():
                child.destroy()
            if self.vars["wavelet"].get() == "Morlet (EXAFS)":
                self._field(tk, ttk, self.dynamic, 0, "Eta", "eta", "Dimensionless Morlet center angular frequency. It controls the k-R resolution trade-off. The 2005 EXAFS paper uses roughly 4-15; eta=5 with sigma=1 is a practical high-k-resolution start, while eta=15 approaches Fourier-like R resolution.")
                self._field(tk, ttk, self.dynamic, 1, "Sigma", "sigma", "Width of the Gaussian envelope. Larger sigma improves R resolution but broadens localization in k; smaller sigma does the opposite. The paper discusses approximately 0.4-2, with 1 as the standard starting value.")
            else:
                self._field(tk, ttk, self.dynamic, 0, "FFT size", "cauchy_nfft", "Minimum FFT length for the Cauchy calculation. 2048 follows the Larch default. 4096 can give a denser internal R grid but uses more memory and time.", ["1024", "2048", "4096", "8192"], 12)

        def _dk_changed(self):
            self.dk_entry.configure(state="disabled" if self.vars["dk_auto"].get() else "normal")

        def _draw_welcome(self):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#f8fafc")
            ax.axis("off")
            ax.text(0.5, 0.58, "EXAFS Wavelet Studio", ha="center", va="center", fontsize=25, fontweight="bold", color="#0f172a", transform=ax.transAxes)
            ax.text(0.5, 0.49, "Morlet and Cauchy analysis in one scientific workspace", ha="center", va="center", fontsize=12, color="#475569", transform=ax.transAxes)
            ax.text(0.5, 0.39, "A demonstration signal is loading...", ha="center", va="center", fontsize=10, color="#0f766e", transform=ax.transAxes)
            ax.text(0.5, 0.08, COPYRIGHT, ha="center", va="center", fontsize=8, color="#94a3b8", transform=ax.transAxes)
            self.canvas.draw_idle()

        def browse_file(self):
            path = self.filedialog.askopenfilename(
                title="Open EXAFS data",
                filetypes=[("Numeric text", "*.txt *.dat *.chi *.xmu *.csv *.tsv"), ("All files", "*.*")],
            )
            if not path:
                return
            self.vars["file"].set(path)
            try:
                self._load_current_file()
            except Exception as exc:
                self.messagebox.showerror("Could not read data", str(exc))

        def _load_current_file(self):
            path = self.vars["file"].get().strip()
            if not path:
                raise ValueError("Choose an input file, or click 'Use demonstration signal'.")
            k, chi = load_numeric_file(
                path,
                self.vars["delimiter"].get(),
                int(self.vars["skip_rows"].get()),
                int(self.vars["k_column"].get()),
                int(self.vars["chi_column"].get()),
            )
            self.loaded_k, self.loaded_chi = k, chi
            self.source_name = Path(path).name
            self.vars["kmin"].set(f"{max(k[0], 3.0):.5g}")
            self.vars["kmax"].set(f"{min(k[-1], 12.0):.5g}" if k[-1] > 3 else f"{k[-1]:.5g}")
            self.status.set(f"Loaded {k.size} rows, k={k[0]:.4g} to {k[-1]:.4g} A^-1.")

        def load_demo(self):
            self.loaded_k, self.loaded_chi = generate_demo_data()
            self.source_name = "Built-in EXAFS-like demonstration"
            self.vars["file"].set("[built-in demonstration]")
            self.vars["kmin"].set("3.0")
            self.vars["kmax"].set("12.0")
            self.status.set("Demonstration loaded. Calculating a first Morlet map...")
            self.run_analysis()

        def _settings(self) -> TransformSettings:
            try:
                return TransformSettings(
                    wavelet=self.vars["wavelet"].get(),
                    kmin=float(self.vars["kmin"].get()),
                    kmax=float(self.vars["kmax"].get()),
                    rmin=float(self.vars["rmin"].get()),
                    rmax=float(self.vars["rmax"].get()),
                    k_points=int(self.vars["k_points"].get()),
                    r_points=int(self.vars["r_points"].get()),
                    eta=float(self.vars["eta"].get()),
                    sigma=float(self.vars["sigma"].get()),
                    dk_auto=bool(self.vars["dk_auto"].get()),
                    dk_manual=float(self.vars["dk_manual"].get()),
                    kweight=int(self.vars["kweight"].get()),
                    taper=self.vars["taper"].get(),
                    cauchy_nfft=int(self.vars["cauchy_nfft"].get()),
                )
            except ValueError as exc:
                raise ValueError("Every numeric parameter must contain a valid number.") from exc

        def run_analysis(self):
            if self._worker and self._worker.is_alive():
                return
            try:
                if self.loaded_k is None or self.loaded_chi is None:
                    self._load_current_file()
                settings = self._settings()
            except Exception as exc:
                self.messagebox.showerror("Check the parameters", str(exc))
                return
            k = self.loaded_k.copy()
            chi = self.loaded_chi.copy()
            source = self.source_name
            self.run_button.configure(state="disabled")
            self.progress_value.set(1.0)
            self.status.set("Preparing data...")

            def update_progress(value: float, text: str):
                self.after(0, lambda: (self.progress_value.set(100.0 * value), self.status.set(f"{text}: {100.0 * value:.0f}%")))

            def worker():
                try:
                    result = calculate_transform(k, chi, settings, source, update_progress)
                    self.after(0, lambda: self._analysis_finished(result))
                except Exception as exc:
                    details = traceback.format_exc()
                    self.after(0, lambda: self._analysis_failed(exc, details))

            self._worker = threading.Thread(target=worker, daemon=True)
            self._worker.start()

        def _analysis_finished(self, result: TransformResult):
            self.result = result
            self.run_button.configure(state="normal")
            self.progress_value.set(100.0)
            mode = "auto" if result.settings.dk_auto else "manual"
            self.status.set(
                f"Ready - {result.coefficients.shape[1]} x {result.coefficients.shape[0]} map; "
                f"dk={result.integration_dk:.5g} A^-1 ({mode})."
            )
            self.refresh_plot()

        def _analysis_failed(self, exc: Exception, details: str):
            self.run_button.configure(state="normal")
            self.progress_value.set(0.0)
            self.status.set("Analysis stopped. Correct the highlighted issue and try again.")
            print(details, file=sys.stderr)
            self.messagebox.showerror("Analysis could not be completed", str(exc))

        def refresh_plot(self):
            if self.result is None:
                return
            try:
                self._map_axis, self._display_matrix = render_analysis_figure(
                    self.figure,
                    self.result,
                    component=self.vars["component"].get(),
                    colormap=self.vars["colormap"].get(),
                    levels=int(self.vars["levels"].get()),
                    log_scale=bool(self.vars["log_scale"].get()),
                    normalize=bool(self.vars["normalize"].get()),
                    show_contours=bool(self.vars["contours"].get()),
                )
                self.canvas.draw_idle()
            except Exception as exc:
                self.messagebox.showerror("Plot setting is not valid", str(exc))

        def _on_plot_motion(self, event):
            if self.result is None or event.inaxes is not self._map_axis or event.xdata is None or event.ydata is None:
                self.cursor_status.set("")
                return
            ki = int(np.clip(np.searchsorted(self.result.k, event.xdata), 0, self.result.k.size - 1))
            ri = int(np.clip(np.searchsorted(self.result.r, event.ydata), 0, self.result.r.size - 1))
            value = self._display_matrix[ri, ki]
            self.cursor_status.set(f"k {self.result.k[ki]:.3f} A^-1   R {self.result.r[ri]:.3f} A   value {value:.4g}")

        def export_figure(self):
            if self.result is None:
                self.messagebox.showinfo("Nothing to export", "Run an analysis first.")
                return
            path = self.filedialog.asksaveasfilename(
                title="Export analysis figure",
                defaultextension=".png",
                initialfile="EXAFS_wavelet_map.png",
                filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("SVG vector", "*.svg")],
            )
            if path:
                self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor=self.figure.get_facecolor())
                self.status.set(f"Figure saved: {path}")

        def export_data(self):
            if self.result is None:
                self.messagebox.showinfo("Nothing to export", "Run an analysis first.")
                return
            path = self.filedialog.asksaveasfilename(
                title="Export wavelet coefficients",
                defaultextension=".npz",
                initialfile="EXAFS_wavelet_result.npz",
                filetypes=[("Compressed NumPy", "*.npz"), ("Tab-separated text", "*.tsv"), ("CSV text", "*.csv")],
            )
            if path:
                export_transform(path, self.result)
                self.status.set(f"Transform data saved: {path}")

        def show_guide(self):
            text = (
                "QUICK START\n\n"
                "1. Load a two-column k / chi(k) file.\n"
                "2. Keep k weight at 0 if your chosen column is already k-weighted.\n"
                "3. Set a clean measured k range and the R region you want to inspect.\n"
                "4. Start with Morlet eta=5 and sigma=1. Keep Auto dk enabled.\n"
                "5. Run the analysis. Hover over the map for numeric k, R, and intensity.\n\n"
                "READING THE MAP\n\n"
                "Horizontal position localizes oscillatory contributions in k; vertical position is the apparent R coordinate. "
                "The R axis is not phase-shift corrected and therefore is not automatically an exact bond distance. Compare transforms only when preprocessing, weighting, ranges, and normalization are documented consistently."
            )
            self.messagebox.showinfo("EXAFS Wavelet Studio - quick guide", text)

        def show_about(self):
            text = (
                f"{APP_NAME} {APP_VERSION}\n\n"
                f"Scientific concept and interface: {AUTHOR}\n{COPYRIGHT}\n\n"
                "Morlet reference:\nH. Funke, A. C. Scheinost, and M. Chukalina, Phys. Rev. B 71, 094110 (2005).\n"
                "DOI: 10.1103/PhysRevB.71.094110\n\n"
                "Cauchy reference:\nM. Munoz, P. Argoul, and F. Farges, American Mineralogist 88, 694-700 (2003).\n\n"
                "Cauchy implementation adapted from the MIT-licensed xraylarch project (Matthew Newville et al.).\n\n"
                "For research and teaching. Validate preprocessing and scientific interpretation independently."
            )
            self.messagebox.showinfo(f"About {APP_NAME}", text)

        def _close(self):
            self.destroy()

    app = WaveletStudio()
    app.mainloop()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--self-test", action="store_true", help="run numerical smoke tests without opening the GUI")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    return _launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
