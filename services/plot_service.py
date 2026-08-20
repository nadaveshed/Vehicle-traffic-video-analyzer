"""Matplotlib figures - the numeric half of the proof.

Per event: the depth curve with the picked minimum on it.
Per job:   the calibration fit, the agreement histogram, and a timeline of
           every capture.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.geometry import box_size, road_contact_point  # noqa: E402
from core.models import (Calibration, ClosestApproach, Detection,  # noqa: E402
                         VideoMeta)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
})

C_FUSED = "#1f4e79"
C_PICK = "#d62728"
C_NAIVE = "#ff9d00"
C_ZONE = "#ffd166"


class PlotService:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        (self.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- per vehicle --
    def event_plot(self, a: ClosestApproach, meta: VideoMeta) -> Path:
        s = a.series
        t = np.array(s["timestamps"], dtype=float)
        frames = np.array(s["frames"])

        def arr(key):
            return np.array([np.nan if v is None else v for v in s[key]], dtype=float)

        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9, 5.4), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )

        in_zone = np.array(s["in_zone"], dtype=bool)
        if in_zone.any():
            ax.axvspan(t[in_zone].min(), t[in_zone].max(), color=C_ZONE, alpha=0.28,
                       label="inside capture zone", zorder=0)

        for key, style, label in (
            ("ground", ":", "ground plane (contact point)"),
            ("width", "--", "apparent width"),
            ("diag", "-.", "box area"),
        ):
            y = arr(key)
            if np.isfinite(y).any():
                ax.plot(t, y, style, lw=1.0, alpha=0.65, label=label)

        ax.plot(t, arr("fused_smooth"), "-", color=C_FUSED, lw=2.4, label="fused (smoothed)")
        ax.plot(t, arr("fused"), ".", color=C_FUSED, ms=3, alpha=0.45, label="fused (raw)")

        unreliable = arr("unreliable")
        if np.isfinite(unreliable).any():
            ax.plot(t, unreliable, ":", color="#c44", lw=1.4, alpha=0.85,
                    label="box cropped - not measurable")

        t_pick = a.refined_timestamp
        ax.axvline(t_pick, color=C_PICK, lw=1.6)
        ax.plot([t_pick], [a.distance_m], "X", color=C_PICK, ms=15, mec="white", mew=1.4,
                zorder=6, label=f"closest approach  f{a.frame_index}  {a.distance_m:.1f} m")

        if a.naive_area_frame is not None and a.naive_area_frame != a.frame_index:
            t_naive = a.naive_area_frame / meta.fps
            ax.axvline(t_naive, color=C_NAIVE, lw=1.6, ls="--",
                       label=f"naive argmax(area)  f{a.naive_area_frame} "
                             f"({a.naive_area_frame - a.frame_index:+d} frames)")

        ax.set_ylabel("estimated distance to camera  [m]")
        ax.set_title(
            f"track #{a.track_id} ({a.vehicle_type})   "
            f"closest at frame {a.frame_index}, t={a.refined_timestamp:.3f}s, "
            f"{a.distance_m:.1f} m"
        )
        # Distance decreases left to right, so the curve climbs across the axes
        # once the y axis is flipped; the top-left corner is the free one.
        ax.legend(loc="upper left", fontsize=7, framealpha=0.92)
        measurable = arr("fused")
        if np.isfinite(measurable).any():
            lo = float(np.nanmin(measurable))
            hi = float(np.nanmax(measurable))
            ax.set_ylim(hi + 0.15 * (hi - lo) + 0.5, lo - 0.10 * (hi - lo) - 0.5)
        else:
            ax.invert_yaxis()

        clipped = np.array(s["cropped"], dtype=bool)
        ax2.fill_between(t, 0, clipped.astype(float), step="mid", color="#c44",
                         alpha=0.55, label="box cropped by frame border")
        ax2.fill_between(t, 0, in_zone.astype(float) * 0.5, step="mid", color=C_ZONE,
                         alpha=0.8, label="in zone")
        ax2.set_ylim(0, 1.1)
        ax2.set_yticks([])
        ax2.set_xlabel("time [s]")
        ax2.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)

        path = self.out_dir / "plots" / f"track_{a.track_id:03d}.png"
        fig.savefig(path, dpi=118, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------ per job --
    def calibration_plot(
        self, detections: list[Detection], calib: Calibration, meta: VideoMeta
    ) -> Path:
        clean = [d for d in detections if not d.clip.any]
        w = np.array([box_size(d.box)[0] for d in clean])
        yb = np.array([road_contact_point(d.box)[1] for d in clean])

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.scatter(w, yb, s=6, alpha=0.20, color=C_FUSED,
                   label=f"{len(clean)} fully-visible detections")

        kw = np.asarray(calib.knots_w)
        ky = np.asarray(calib.knots_y)
        ax.plot(kw, ky, "-o", color=C_PICK, lw=2.2, ms=4,
                label=f"measured camera curve   R2 = {calib.r_squared:.3f}")
        if calib.invert_from > 0:
            edge = kw[calib.invert_from]
            ax.axvspan(kw.min(), edge, color="#999", alpha=0.16)
            ax.text((kw.min() + edge) / 2, meta.height * 0.80,
                    "too far to read depth" + "\n" + "from the contact row",
                    fontsize=8, color="#555", va="center", ha="center",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", alpha=0.9))
        ax.axhline(calib.horizon_y, color="#666", ls="--", lw=1.2,
                   label=f"effective horizon  y = {calib.horizon_y:.0f} px")
        ax.set_xlabel("bounding-box width  [px]")
        ax.set_ylabel("box bottom edge (road contact point)  [px]")
        ax.set_title(
            "Self-calibration: apparent width against contact row, measured rather\n"
            f"than assumed. Near-field slope puts the camera at "
            f"{calib.camera_height_m:.1f} m."
        )
        ax.set_ylim(meta.height, 0)
        ax.legend(fontsize=8)
        path = self.out_dir / "plots" / "calibration_fit.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    def agreement_plot(self, approaches: list[ClosestApproach], meta: VideoMeta) -> Path:
        spread = np.array([a.agreement_frames for a in approaches], dtype=float)
        naive = np.array([
            (a.naive_area_frame - a.frame_index) for a in approaches
            if a.naive_area_frame is not None
        ], dtype=float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # A handful of distant vehicles disagree wildly and would flatten the
        # bulk of the distribution into invisibility, so the tails are pooled
        # into an overflow bin rather than dropped.
        cap = 8
        over = int((spread > cap).sum())
        ax1.hist(np.clip(spread, 0, cap + 1), bins=np.arange(-0.5, cap + 2.5),
                 color=C_FUSED, alpha=0.88, edgecolor="white")
        ax1.set_xticks(list(range(0, cap + 2)))
        ax1.set_xticklabels([str(i) for i in range(cap + 1)] + [f">{cap}"])
        ax1.set_xlabel("spread between the 3 independent estimates  [frames]")
        ax1.set_ylabel("vehicles")
        ax1.set_title(f"Cross-metric agreement" + "\n" +
                      f"median {np.median(spread):.0f}, "
                      f"{(spread <= 1).mean() * 100:.0f}% within 1 frame"
                      + (f", {over} beyond {cap}" if over else ""))

        ncap = 10
        nover = int((np.abs(naive) > ncap).sum()) if len(naive) else 0
        if len(naive):
            ax2.hist(np.clip(naive, -ncap - 1, ncap + 1),
                     bins=np.arange(-ncap - 1.5, ncap + 2.5),
                     color=C_NAIVE, alpha=0.92, edgecolor="white")
            ax2.set_xticks([-ncap - 1, -ncap // 2, 0, ncap // 2, ncap + 1])
            ax2.set_xticklabels([f"<-{ncap}", str(-ncap // 2), "0",
                                 str(ncap // 2), f">{ncap}"])
        ax2.axvline(0, color=C_PICK, lw=2)
        ax2.set_xlabel("naive argmax(area) frame  -  our frame")
        ax2.set_ylabel("vehicles")
        disagreed = (np.abs(naive) > 0).mean() * 100 if len(naive) else 0.0
        ax2.set_title("Error of the naive baseline" + "\n" +
                      f"disagrees on {disagreed:.0f}% of vehicles"
                      + (f", {nover} by more than {ncap} frames" if nover else ""))
        fig.tight_layout()
        path = self.out_dir / "plots" / "validation_agreement.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    def timeline_plot(self, approaches: list[ClosestApproach], meta: VideoMeta) -> Path:
        approaches = sorted(approaches, key=lambda a: a.first_frame)
        fig, ax = plt.subplots(figsize=(11, max(3.2, 0.24 * len(approaches) + 1.4)))
        for row, a in enumerate(approaches):
            t0, t1 = a.first_frame / meta.fps, a.last_frame / meta.fps
            ax.plot([t0, t1], [row, row], lw=3, color="#b8cbe0", solid_capstyle="butt")
            if a.entered_zone:
                ax.plot([a.zone_entry_frame / meta.fps, a.zone_exit_frame / meta.fps],
                        [row, row], lw=6, color=C_ZONE, solid_capstyle="butt", zorder=2)
            ax.plot([a.refined_timestamp], [row], "X", color=C_PICK, ms=8, zorder=4)
        ax.set_yticks(range(len(approaches)))
        ax.set_yticklabels([f"#{a.track_id} {a.vehicle_type}" for a in approaches], fontsize=7)
        ax.set_xlabel("time [s]")
        ax.set_xlim(0, meta.duration)
        ax.set_title("Every tracked vehicle: lifetime (grey), dwell inside the capture "
                     "zone (yellow), closest approach (✕)")
        path = self.out_dir / "plots" / "timeline.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
