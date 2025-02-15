import os, glob, re
from utils.config import check_and_mkdir_if_neccasiry


def vvt_to_text(vtt_file_path):
    text_lines = []
    with open(vtt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            # Skip lines that are empty or contain time codes or headers
            if not line.strip() or '-->' in line or line.startswith('WEBVTT') or 'Kind:' in line or 'Language:' in line:
                continue
            text_lines.append(line.strip())
    return '\n'.join(text_lines)


def preprocess_all_vtt_files(root_dir, output_dir):
    vvt_files = glob.glob(os.path.join(root_dir, '*.vtt'))
    
    for vvt_file in vvt_files:
        text = vvt_to_text(vvt_file)
        output_dir = check_and_mkdir_if_neccasiry(output_dir)
        text_file = os.path.join(output_dir, os.path.basename(vvt_file).replace('.vtt', '.txt'))
        with open(os.path.join(root_dir, '..', 'working', text_file), 'w', encoding='utf-8') as file:
            file.write(text)
