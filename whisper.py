#!/usr/bin/env python3

import os.path as path
import time
from argparse import ArgumentParser

from faster_whisper import WhisperModel


def sec2time(sec):
    sec = int(sec)
    return f"{sec//3600:02d}:{sec//60%60:02d}:{sec%60:02d}"


parser = ArgumentParser(description="audio path and settings")
parser.add_argument(
    "audios", type=str, nargs="+", help="audio path(s) separated by spaces"
)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    help="device for computation: auto (default), cpu, cuda",
    default="auto",
)
parser.add_argument(
    "-f",
    "--format",
    type=str,
    default=".txt",
    help="output file format (default: txt)",
)
parser.add_argument(
    "-i",
    "--interval",
    type=int,
    default=60,
    help="add timestamps by interval (unit: second, default: 60)",
)
parser.add_argument(
    "-l",
    "--language",
    type=str,
    help="set language (default: None (auto-detected))",
    default=None,
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="model size: tiny, tiny.en, base, base.en, small, "
    "small.en, medium, medium.en, large-v1, or large-v2 (default)\n"
    "or a path to a converted model directory",
    default="large-v2",
)
parser.add_argument(
    "-o",
    "--outdir",
    type=str,
    default=None,
    help="directory to store the output (default: audio dir)",
)
parser.add_argument(
    "-z",
    "--zhtype",
    action="store_true",
    default=False,
    help="basic Chinese typesetting",
)

args = parser.parse_args()
audios, device, format, interval, language, model_size_or_path, outdir, zhtype = (
    args.audios,
    args.device,
    args.format,
    args.interval,
    args.language,
    args.model,
    args.outdir,
    args.zhtype,
)
if format[0] != ".":
    format = "." + format

model = WhisperModel(
    model_size_or_path=model_size_or_path, device=device, compute_type="int8_float16"
)

for audio in audios:
    start = time.perf_counter()
    quote = [None]

    def zhtypeset(char):
        match char:
            case "(":
                return "（"
            case ")":
                return "）"
            case "[":
                return "【"
            case "]":
                return "】"
            case ",":
                return "，"
            case ".":
                return "。"
            case "!":
                return "！"
            case "?":
                return "？"
            case ":":
                return "："
            case '"':
                if quote[-1] == '"':
                    list.pop(quote)
                    return "”"
                list.append(quote, '"')
                return "“"
            case "'":
                if quote[-1] == "'":
                    list.pop(quote)
                    return "’"
                list.append(quote, "'")
                return "‘"
            case _:
                return char

    audio = audio.strip("'\"")
    dir, file = path.split(audio)
    base, _ = path.splitext(file)
    if outdir == None:
        output = path.join(dir, base + format)
    else:
        output = path.join(outdir, base + format)
    f = open(output, "w", encoding="utf-8")
    segments, info = model.transcribe(
        audio=audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        initial_prompt="Hello, world. 请使用标点，谢谢。",
    )
    stamp = -interval
    print(f"transcribing {audio}")
    f.writelines(f"--Transcription of {audio}--")
    for segment in segments:
        print(
            f"\raudio: {sec2time(segment.end)}, time used: {sec2time(time.perf_counter() - start)}",
            end="",
        )
        if int(segment.start) >= stamp + interval:
            stamp = int(segment.start) // interval * interval
            f.writelines(f"\n{sec2time(stamp)}\n")
        if zhtype and info.language == "zh":
            f.writelines(map(zhtypeset, segment.text))
        else:
            f.writelines(segment.text)
        if segment.text[-1] not in "'’\"”.。,，!！?？:：)）]】}、":
            f.writelines(" ")
    print(f"\ncomplete")
    f.close()
