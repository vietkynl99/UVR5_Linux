#!/usr/bin/env python3
"""Simple CLI wrapper for UVR5 inference.

Currently supports VR (inference_v5) with the same model files used by the GUI.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import types

import tkinter as tk

import inference_v5
from pydub import AudioSegment
from pydub.utils import mediainfo


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VR_MODELS_DIR = os.path.join(MODELS_DIR, "Main_Models")


class ConsoleText:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def write(self, message: str):
        if self.quiet:
            return
        sys.stdout.write(message)
        sys.stdout.flush()

    def clear(self):
        # GUI clears the text widget; no-op for CLI.
        return


class DummyButton:
    def configure(self, **_kwargs):
        return


class ProgressVar:
    def __init__(self):
        self.value = 0

    def set(self, value):
        self.value = value


def list_vr_models():
    if not os.path.isdir(VR_MODELS_DIR):
        return []
    return sorted([f for f in os.listdir(VR_MODELS_DIR) if f.endswith(".pth")])


def resolve_vr_model(model_arg: str) -> str:
    """Resolve model argument to a file path."""
    if os.path.isfile(model_arg):
        return os.path.abspath(model_arg)
    if not model_arg.endswith(".pth"):
        model_arg = model_arg + ".pth"
    candidate = os.path.join(VR_MODELS_DIR, model_arg)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    return ""


def _get_audio_duration_seconds(path: str):
    try:
        info = mediainfo(path)
        duration = info.get("duration")
        if duration:
            return float(duration)
    except Exception:
        pass
    try:
        audio = AudioSegment.from_file(path)
        return len(audio) / 1000.0
    except Exception:
        return None


def _split_audio_if_needed(path: str, max_seconds: int, work_root: str, logger):
    duration = _get_audio_duration_seconds(path)
    if duration is None or duration <= max_seconds:
        return [path], None

    base_name = os.path.splitext(os.path.basename(path))[0]
    split_prefix = f"uvr_cli_split_{base_name}_"

    # Reuse existing split folder if present.
    if os.path.isdir(work_root):
        candidates = []
        for name in os.listdir(work_root):
            if not name.startswith(split_prefix):
                continue
            dir_path = os.path.join(work_root, name)
            if not os.path.isdir(dir_path):
                continue
            wavs = sorted(
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(".wav")
            )
            if wavs:
                try:
                    mtime = os.path.getmtime(dir_path)
                except Exception:
                    mtime = 0
                candidates.append((mtime, dir_path, wavs))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, split_dir, wavs = candidates[0]
            logger.write(
                f"Reusing existing split folder '{os.path.basename(split_dir)}' "
                f"with {len(wavs)} chunk(s).\n"
            )
            return wavs, split_dir

    split_dir = tempfile.mkdtemp(prefix=f"uvr_cli_split_{base_name}_", dir=work_root)
    chunk_pattern = os.path.join(split_dir, f"{base_name}_part_%03d.wav")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            path,
            "-f",
            "segment",
            "-segment_time",
            str(max_seconds),
            "-c:a",
            "pcm_s16le",
            chunk_pattern,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            chunks = sorted(
                os.path.join(split_dir, f)
                for f in os.listdir(split_dir)
                if f.endswith(".wav")
            )
            if chunks:
                logger.write(f"Split '{os.path.basename(path)}' into {len(chunks)} chunk(s).\n")
                return chunks, split_dir
        logger.write("FFmpeg split failed, falling back to in-memory split.\n")

    try:
        audio = AudioSegment.from_file(path)
        chunks = []
        for idx, start_ms in enumerate(range(0, len(audio), max_seconds * 1000), start=1):
            chunk = audio[start_ms : start_ms + max_seconds * 1000]
            chunk_path = os.path.join(split_dir, f"{base_name}_part_{idx:03d}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        logger.write(f"Split '{os.path.basename(path)}' into {len(chunks)} chunk(s).\n")
        return chunks, split_dir
    except Exception:
        return [path], None


def _concat_audio_files(files, output_path, save_format, mp3_bitrate, logger):
    if not files:
        return False

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        list_file = None
        try:
            list_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, dir=os.path.dirname(output_path)
            )
            for path in files:
                escaped = path.replace("'", "'\\''")
                list_file.write(f"file '{escaped}'\n")
            list_file.close()

            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file.name,
                "-c",
                "copy",
                output_path,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                return True
        finally:
            if list_file and os.path.isfile(list_file.name):
                try:
                    os.remove(list_file.name)
                except Exception:
                    pass
        logger.write("FFmpeg concat failed, falling back to re-encode.\n")

    try:
        combined = AudioSegment.empty()
        for path in files:
            combined += AudioSegment.from_file(path)
        export_kwargs = {"format": save_format.lower()}
        if save_format.lower() == "mp3":
            export_kwargs["bitrate"] = mp3_bitrate
        combined.export(output_path, **export_kwargs)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="UVR5 CLI wrapper (VR backend).")
    parser.add_argument("-i", "--input", nargs="+", required=False, help="Input audio files")
    parser.add_argument("-o", "--output", required=False, help="Output directory")
    parser.add_argument("--backend", default="vr", choices=["vr"], help="Backend (currently vr only)")
    parser.add_argument("--model", default="MGM_MAIN_v4.pth", help="VR model name or path")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    parser.add_argument("--list-models", action="store_true", help="List models for selected backend")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm prompts")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index (use -1 for CPU)")
    parser.add_argument("--agg", type=int, default=10, help="Aggression (VR)")
    parser.add_argument("--window-size", type=int, default=512, help="Window size (VR)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap (VR)")
    parser.add_argument("--shifts", type=int, default=2, help="Shifts (VR)")
    parser.add_argument("--split-mode", action="store_true", help="Enable split mode (VR)")
    parser.add_argument("--no-split-mode", dest="split_mode", action="store_false", help="Disable split mode")
    parser.set_defaults(split_mode=True)
    parser.add_argument("--tta", action="store_true", help="Enable TTA (VR)")
    parser.add_argument("--postprocess", action="store_true", help="Enable postprocess (VR)")
    parser.add_argument("--normalize", action="store_true", help="Normalize output")
    parser.add_argument("--save-format", default="Mp3", choices=["Wav", "Flac", "Mp3"], help="Output format")
    parser.add_argument("--wavtype", default="PCM_16", help="WAV subtype (e.g., PCM_16)")
    parser.add_argument("--flactype", default="PCM_16", help="FLAC subtype (e.g., PCM_16)")
    parser.add_argument("--mp3bit", default="320k", help="MP3 bitrate (e.g., 320k)")
    parser.add_argument("--model-params", default="Auto", help="Model params (Auto or specific)")
    parser.add_argument("--max-chunk-seconds", type=int, default=1800, help="Max chunk length in seconds when splitting long inputs",)
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument(
        "--output-name",
        required=False,
        help=(
            "Custom output filename template (without directory). "
            "Available fields: {orig_index}, {orig_base}, {seq}, {chunk_base}, "
            "{suffix}, {ext}. If {ext} is omitted, the extension is appended."
        ),
    )
    stem_group = parser.add_mutually_exclusive_group()
    stem_group.add_argument(
        "--instrumental-only",
        action="store_true",
        default=True,
        help="Export only instrumental stems (default)",
    )
    stem_group.add_argument(
        "--both",
        action="store_true",
        help="Export both instrumental and vocals",
    )

    args = parser.parse_args()

    if args.list_backends:
        print("vr")
        return 0

    if args.list_models:
        if args.backend == "vr":
            for name in list_vr_models():
                print(name)
        return 0

    if not args.input:
        parser.error("--input is required unless using --list-* options")

    if args.backend != "vr":
        print("Only 'vr' backend is supported in this CLI wrapper.", file=sys.stderr)
        return 2

    model_path = resolve_vr_model(args.model)
    if not model_path:
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 2

    if args.output:
        output_dir = args.output
    else:
        input_dirs = {os.path.dirname(os.path.abspath(p)) for p in args.input}
        if len(input_dirs) != 1:
            print("Multiple input directories detected. Please provide --output.", file=sys.stderr)
            return 2
        output_dir = input_dirs.pop()

    os.makedirs(output_dir, exist_ok=True)

    text_widget = ConsoleText(quiet=args.quiet)
    button_widget = DummyButton()
    progress_var = ProgressVar()

    save_ext = {
        "Wav": "wav",
        "Flac": "flac",
        "Mp3": "mp3",
    }[args.save_format]

    expected_suffixes = ["(Instrumental)"]
    if args.both:
        expected_suffixes.append("(Vocals)")

    max_chunk_seconds = args.max_chunk_seconds
    expanded_inputs = []
    split_plan = []
    for orig_index, input_path in enumerate(args.input, start=1):
        chunks, split_dir = _split_audio_if_needed(
            input_path, max_chunk_seconds, output_dir, text_widget
        )
        split_plan.append(
            {
                "orig_index": orig_index,
                "orig_path": input_path,
                "orig_base": os.path.splitext(os.path.basename(input_path))[0],
                "chunks": chunks,
                "split_dir": split_dir,
            }
        )
        for chunk_path in chunks:
            chunk_base = os.path.splitext(os.path.basename(chunk_path))[0]
            expanded_inputs.append(
                {
                    "orig_index": orig_index,
                    "chunk_path": chunk_path,
                    "chunk_base": chunk_base,
                    "seq": len(expanded_inputs) + 1,
                }
            )

    def askyesno(title, message):
        if args.yes:
            return True
        prompt = f"{title}: {message} [y/N]: "
        try:
            return input(prompt).strip().lower().startswith("y")
        except EOFError:
            return False

    def showerror(master=None, title="Error", message=""):
        print(f"{title}: {message}", file=sys.stderr)

    tk.messagebox = types.SimpleNamespace(askyesno=askyesno, showerror=showerror)

    output_name_template = args.output_name

    def _render_output_name(template, context, fallback_name):
        if not template:
            return fallback_name
        try:
            name = template.format(**context)
        except KeyError as exc:
            raise ValueError(f"Unknown placeholder in --output-name: {exc}") from exc
        if "{ext}" not in template and not name.lower().endswith(f".{context['ext'].lower()}"):
            name = f"{name}.{context['ext']}"
        return name

    def _produced_outputs(seq, chunk_base):
        base_name = os.path.join(output_dir, f"{seq}_{chunk_base}")
        return [
            f"{base_name}_{suffix}.{save_ext}"
            for suffix in expected_suffixes
        ]

    def _target_output_path(orig_index, orig_base, seq, chunk_base, suffix, is_split):
        if is_split:
            fallback = f"{orig_index}_{orig_base}_{suffix}.{save_ext}"
        else:
            fallback = f"{seq}_{chunk_base}_{suffix}.{save_ext}"
        context = {
            "orig_index": orig_index,
            "orig_base": orig_base,
            "seq": seq,
            "chunk_base": chunk_base,
            "suffix": suffix,
            "ext": save_ext,
        }
        name = _render_output_name(output_name_template, context, fallback)
        return os.path.join(output_dir, name)

    def _expected_outputs_for_resume(entry, plan):
        if not output_name_template:
            return _produced_outputs(entry["seq"], entry["chunk_base"])
        is_split = len(plan["chunks"]) > 1
        return [
            _target_output_path(
                plan["orig_index"],
                plan["orig_base"],
                entry["seq"],
                entry["chunk_base"],
                suffix,
                is_split,
            )
            for suffix in expected_suffixes
        ]

    def _safe_rename(src, dst, logger):
        if not os.path.isfile(src) or src == dst:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(dst):
            logger.write(f"Target exists, keeping existing file: {dst}\n")
            return
        try:
            os.replace(src, dst)
        except Exception:
            logger.write(f"Failed to rename '{src}' -> '{dst}'.\n")

    resume_enabled = True
    to_process = expanded_inputs
    if resume_enabled:
        first_missing_seq = None
        plan_by_orig = {p["orig_index"]: p for p in split_plan}
        for entry in expanded_inputs:
            plan = plan_by_orig[entry["orig_index"]]
            outputs = _expected_outputs_for_resume(entry, plan)
            if not all(os.path.isfile(p) for p in outputs):
                first_missing_seq = entry["seq"]
                break

        if first_missing_seq is None:
            to_process = []
            text_widget.write("Resume enabled: all chunks already processed. Skipping inference.\n")
        else:
            to_process = [e for e in expanded_inputs if e["seq"] >= first_missing_seq]
            if first_missing_seq > 1:
                text_widget.write(
                    f"Resume enabled: starting from chunk {first_missing_seq}.\n"
                )
            # Remove any outputs for later chunks so they get re-generated.
            for entry in expanded_inputs:
                if entry["seq"] < first_missing_seq:
                    continue
                plan = plan_by_orig[entry["orig_index"]]
                for path in _expected_outputs_for_resume(entry, plan):
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass

    if not to_process:
        pass
    else:
        kwargs = {
            "agg": args.agg,
            "gpu": args.gpu,
            "input_paths": [e["chunk_path"] for e in to_process],
            "instrumentalModel": model_path,
            "export_path": os.path.abspath(output_dir),
            "ModelParams": args.model_params,
            "saveFormat": args.save_format,
            "wavtype": args.wavtype,
            "flactype": args.flactype,
            "mp3bit": args.mp3bit,
            "overlap": args.overlap,
            "shifts": args.shifts,
            "split_mode": args.split_mode,
            "tta": args.tta,
            "postprocess": args.postprocess,
            "normalize": args.normalize,
            "segment": "None",
            "settest": False,
            "inst_only": (not args.both),
            "voc_only": False,
            "useModel": "instrumental",
            "demucsmodelVR": False,
            "demucsmodel_sel_VR": "UVR_Demucs_Model_1",
            "modelFolder": False,
            "output_image": False,
            "window_size": args.window_size,
        }

        inference_v5.main(None, text_widget, button_widget, progress_var, **kwargs)

        if len(to_process) != len(expanded_inputs):
            for run_idx, entry in enumerate(to_process, start=1):
                produced = _expected_outputs(run_idx, entry["chunk_base"])
                target = _expected_outputs(entry["seq"], entry["chunk_base"])
                for src, dst in zip(produced, target):
                    if not os.path.isfile(src):
                        continue
                    if os.path.isfile(dst):
                        try:
                            os.remove(src)
                        except Exception:
                            pass
                        continue
                    try:
                        os.replace(src, dst)
                    except Exception:
                        pass

        if output_name_template:
            plan_by_orig = {p["orig_index"]: p for p in split_plan}
            for entry in expanded_inputs:
                plan = plan_by_orig[entry["orig_index"]]
                if len(plan["chunks"]) > 1:
                    continue
                for suffix in expected_suffixes:
                    src = _produced_outputs(entry["seq"], entry["chunk_base"])[
                        expected_suffixes.index(suffix)
                    ]
                    dst = _target_output_path(
                        plan["orig_index"],
                        plan["orig_base"],
                        entry["seq"],
                        entry["chunk_base"],
                        suffix,
                        False,
                    )
                    _safe_rename(src, dst, text_widget)

    has_splits = any(len(p["chunks"]) > 1 for p in split_plan)
    if has_splits:
        outputs_by_orig = {}
        for plan in split_plan:
            outputs_by_orig[plan["orig_index"]] = {
                "(Instrumental)": [],
                "(Vocals)": [],
            }

        for file_num, entry in enumerate(expanded_inputs, start=1):
            chunk_base = entry["chunk_base"]
            base_name = os.path.join(output_dir, f"{file_num}_{chunk_base}")
            for suffix in ("(Instrumental)", "(Vocals)"):
                candidate = f"{base_name}_{suffix}.{save_ext}"
                if os.path.isfile(candidate):
                    outputs_by_orig[entry["orig_index"]][suffix].append(candidate)

        for plan in split_plan:
            if len(plan["chunks"]) <= 1:
                continue
            orig_base = plan["orig_base"]
            orig_index = plan["orig_index"]
            for suffix, parts in outputs_by_orig[orig_index].items():
                if not parts:
                    continue
                if len(parts) != len(plan["chunks"]):
                    text_widget.write(
                        f"Chunk count mismatch for '{orig_base}' {suffix}. "
                        f"Expected {len(plan['chunks'])}, found {len(parts)}. "
                        "Skipping merge.\n"
                    )
                    continue
                final_path = os.path.join(
                    output_dir, f"{orig_index}_{orig_base}_{suffix}.{save_ext}"
                )
                ok = _concat_audio_files(
                    parts, final_path, args.save_format, args.mp3bit, text_widget
                )
                if ok:
                    for part in parts:
                        try:
                            os.remove(part)
                        except Exception:
                            pass
                    if output_name_template:
                        target_path = _target_output_path(
                            orig_index,
                            orig_base,
                            orig_index,
                            orig_base,
                            suffix,
                            True,
                        )
                        _safe_rename(final_path, target_path, text_widget)
                else:
                    text_widget.write(
                        f"Failed to merge chunks for '{orig_base}' {suffix}. Keeping chunk outputs.\n"
                    )

        for plan in split_plan:
            if plan["split_dir"] and os.path.isdir(plan["split_dir"]):
                try:
                    shutil.rmtree(plan["split_dir"])
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
