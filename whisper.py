#!/usr/bin/env python3

from faster_whisper import WhisperModel
import time
import argparse
import os.path as path

def sec2time(sec):
    sec = int(sec)
    return f'{sec//3600:02d}:{sec//60%60:02d}:{sec%60:02d}'

parser = argparse.ArgumentParser(description='audio path and settings')
parser.add_argument('audio_path',
                    type=str,
                    nargs='+',
                    help='split different paths with spaces')
parser.add_argument('-i', '--interval',
                    type=int,
                    help='add timestamps by interval (unit: second, default: 60)',
                    default=60)
parser.add_argument('-o', '--outdir', 
                    type=str,
                    help='directory to store the output (default: audio dir)',
                    default=None)
parser.add_argument('-f', '--format',
                    type=str,
                    help='output file format (default: txt)',
                    default='.txt')
parser.add_argument('-m', '--model',
                    type=str,
                    help='size of the model to use: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, or large-v2 (default); '
                        'or a path to a converted model directory',
                    default='large-v2')
parser.add_argument('-l', '--language',
                    type=str,
                    help='set language (default: auto-detected)',
                    default=None)
parser.add_argument('-z', '--zhpunc',
                    type=bool,
                    help='True or False (default): use Chinese punctuation (full-width) if the language is Chinese',
                    default=False)
parser.add_argument('-d', '--device',
                    type=str,
                    help='device for computation: cpu, cuda, auto (default)',
                    default='auto')
args = parser.parse_args()
audios, interval, outdir, format, model, lang, zhpunc, device = args.audio_path, args.interval, args.outdir, args.format, args.model, args.language, args.zhpunc, args.device
if format[0] != '.':
    format = '.' + format

quote = [None,]
def zh_punc(char):
    match char:
        case '(':   return '（'
        case ')':   return '）'
        case '[':   return '【'
        case ']':   return '】'
        case ',':   return '，'
        case '.':   return '。'
        case '!':   return '！'
        case '?':   return '？'
        case ':':   return '：'
        case '"':
            if quote[-1] == '"':
                list.pop(quote)
                return '”'
            list.append(quote, '"')
            return '“'
        case "'":
            if quote[-1] == "'":
                list.pop(quote)
                return '’'
            list.append(quote, "'")
            return '‘'
        case _:     return char

for audio in audios:
    start = time.perf_counter()
    model = WhisperModel(model_size_or_path=model,
                        device=device,
                        compute_type='int8_float16')
    audio = audio.strip('\'"')
    dir, file = path.split(audio)
    base, _ = path.splitext(file)
    if outdir == None:
        output = path.join(dir, base + format)
    else:
        output = path.join(outdir, base + format)
    f = open(output, 'w', encoding='utf-8')
    segments, info = model.transcribe(audio=audio, 
                                      word_timestamps=True, 
                                      vad_filter=True,
                                      language=lang)
    lang = info.language
    stamp = -interval
    print(f'transcribing {audio}')
    f.writelines(f'--Transcription of {audio}--')
    for segment in segments:
        print(f'\raudio: {sec2time(segment.end)}, time used: {sec2time(time.perf_counter() - start)}', end='')
        if int(segment.start) >= stamp + interval:
            stamp = int(segment.start)//interval*interval
            f.writelines(f'\n{sec2time(stamp)}\n')
        if lang == 'zh' and zhpunc:
            f.writelines(map(zh_punc, segment.text))
        else:
            f.writelines(segment.text)
        if segment.text[-1] not in '"\'.。,，!！?？:：’”)）]】}、':
            f.writelines(' ')
    print(f'\ncomplete')
    f.close()
