# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gasket_inspection.config import load_config, resolve_project_path  # noqa: E402


STATUS_COLORS = {
    "OK": ("#0d7a3a", "#ffffff"),
    "NG": ("#c62828", "#ffffff"),
    "ERROR": ("#ef6c00", "#ffffff"),
}


KOREAN_FONT_CANDIDATES = (
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "NanumGothic",
    "Malgun Gothic",
    "Apple SD Gothic Neo",
    "Arial Unicode MS",
)


def select_korean_font(root: tk.Tk) -> str:
    """Return an installed font family that can render Korean text."""
    installed = {family.casefold(): family for family in tkfont.families(root)}
    for candidate in KOREAN_FONT_CANDIDATES:
        if match := installed.get(candidate.casefold()):
            return match

    # Tk's platform default is safer than forcing a generic family with no
    # Hangul glyphs. Modern desktop environments usually provide fallback.
    return str(tkfont.nametofont("TkDefaultFont", root=root).cget("family"))


class InspectionDisplay:
    def __init__(self, root: tk.Tk, inspections_path: Path, poll_ms: int) -> None:
        self.root = root
        self.inspections_path = inspections_path
        self.poll_ms = poll_ms
        self.last_signature: tuple[int, int] | None = None
        self.font_family = select_korean_font(root)

        style = ttk.Style(root)
        style.configure("Treeview", font=(self.font_family, 11))
        style.configure("Treeview.Heading", font=(self.font_family, 11, "bold"))

        root.title("가스켓 실시간 검사 결과")
        root.geometry("1000x700")
        root.minsize(760, 560)
        root.configure(bg="#101820")

        self.header = tk.Label(
            root,
            text="실시간 제품 검사",
            font=(self.font_family, 24, "bold"),
            fg="#ffffff",
            bg="#101820",
        )
        self.header.pack(pady=(24, 8))

        self.object_label = tk.Label(
            root,
            text="검사 결과 대기 중",
            font=(self.font_family, 22, "bold"),
            fg="#d9e2ec",
            bg="#101820",
        )
        self.object_label.pack(pady=(10, 8))

        self.status_label = tk.Label(
            root,
            text="WAIT",
            font=(self.font_family, 72, "bold"),
            fg="#ffffff",
            bg="#455a64",
            width=10,
            pady=16,
        )
        self.status_label.pack(pady=10)

        self.defect_label = tk.Label(
            root,
            text="네 카메라의 판정 완료를 기다리고 있습니다.",
            font=(self.font_family, 18, "bold"),
            fg="#ffffff",
            bg="#101820",
            wraplength=900,
            justify="center",
        )
        self.defect_label.pack(pady=(12, 20))

        history_frame = tk.Frame(root, bg="#101820")
        history_frame.pack(fill="both", expand=True, padx=28, pady=(0, 14))
        tk.Label(
            history_frame,
            text="최근 검사 기록",
            font=(self.font_family, 14, "bold"),
            fg="#d9e2ec",
            bg="#101820",
        ).pack(anchor="w", pady=(0, 6))

        self.history = ttk.Treeview(
            history_frame,
            columns=("number", "cycle", "status", "defects"),
            show="headings",
            height=8,
        )
        self.history.heading("number", text="순번")
        self.history.heading("cycle", text="제품")
        self.history.heading("status", text="판정")
        self.history.heading("defects", text="검출 불량")
        self.history.column("number", width=80, anchor="center", stretch=False)
        self.history.column("cycle", width=180, anchor="center", stretch=False)
        self.history.column("status", width=90, anchor="center", stretch=False)
        self.history.column("defects", width=500, anchor="w")
        self.history.pack(fill="both", expand=True)

        self.footer = tk.Label(
            root,
            text=f"감시 파일: {inspections_path}",
            font=(self.font_family, 10),
            fg="#9fb3c8",
            bg="#101820",
        )
        self.footer.pack(pady=(0, 12))

        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", lambda _event: root.attributes("-fullscreen", False))
        self._poll()

    def _toggle_fullscreen(self, _event: tk.Event) -> None:
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))

    def _poll(self) -> None:
        try:
            stat = self.inspections_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != self.last_signature:
                with self.inspections_path.open("r", encoding="utf-8") as handle:
                    document = json.load(handle)
                inspections = document.get("inspections", [])
                if not isinstance(inspections, list):
                    raise ValueError("inspections 항목이 배열이 아닙니다.")
                self.last_signature = signature
                self._render(inspections)
        except FileNotFoundError:
            self.footer.configure(text="inspections.json 생성을 기다리는 중입니다.")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.footer.configure(text=f"검사 기록 읽기 오류: {exc}", fg="#ffb4a9")
        finally:
            self.root.after(self.poll_ms, self._poll)

    def _render(self, inspections: list[object]) -> None:
        valid = [item for item in inspections if isinstance(item, dict)]
        self.history.delete(*self.history.get_children())
        for item in reversed(valid[-10:]):
            decision = item.get("decision", {})
            names = decision.get("defect_names_ko", [])
            self.history.insert(
                "",
                "end",
                values=(
                    item.get("inspection_number", "-"),
                    item.get("cycle_id", "-"),
                    decision.get("status", "-"),
                    ", ".join(names) if names else "양품",
                ),
            )

        if not valid:
            self.footer.configure(
                text=f"감시 중 · 누적 0건 · {self.inspections_path}", fg="#9fb3c8"
            )
            return

        latest = valid[-1]
        decision = latest.get("decision", {})
        status = str(decision.get("status", "ERROR"))
        background, foreground = STATUS_COLORS.get(status, STATUS_COLORS["ERROR"])
        number = latest.get("inspection_number", "-")
        cycle_id = latest.get("cycle_id", "-")
        self.object_label.configure(text=f"검사 #{number} · {cycle_id}")
        self.status_label.configure(text=status, bg=background, fg=foreground)

        if status == "NG":
            details = []
            for defect in decision.get("detected_defects", []):
                cameras = ", ".join(str(value) for value in defect.get("detected_cameras", []))
                details.append(f"{defect.get('defect_name_ko', '불량')} (카메라 {cameras})")
            self.defect_label.configure(text="검출 불량: " + " · ".join(details))
        elif status == "OK":
            self.defect_label.configure(text="검출된 불량이 없습니다.")
        else:
            self.defect_label.configure(text="일부 카메라 판별 중 오류가 발생했습니다.")
        self.footer.configure(
            text=f"실시간 감시 중 · 누적 {len(valid)}건 · F11 전체화면 · Esc 해제",
            fg="#9fb3c8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="제품별 통합 검사 결과 실시간 GUI")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--poll-ms", type=int, default=250)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()
    if args.poll_ms < 50:
        parser.error("--poll-ms는 50 이상이어야 합니다.")

    cfg = load_config(args.config)
    inspections_path = resolve_project_path(
        cfg, cfg["realtime"].get("inspections_file", "runtime/state/inspections.json")
    )
    root = tk.Tk()
    if args.fullscreen:
        root.attributes("-fullscreen", True)
    InspectionDisplay(root, inspections_path, args.poll_ms)
    root.mainloop()


if __name__ == "__main__":
    main()
