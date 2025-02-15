import os
import yt_dlp 

from pytube import Playlist
from utils.config import ROOT_DIR


def download_subtitles(url, output_dir):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'subtitlesformat': 'vtt',
        'outtmpl': output_dir,
        'subtitleslangs': ['tr_sdh', 'tr', 'tr-tr'], 
        'writesubtitles': True,
        # 'listsubtitles': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_playlistCC(url):
    playlist = Playlist(url)
    print(f'There are {len(playlist.videos)} videos in the playlist.')
    
    for video in playlist.videos:
        print(f'Current is {video.title}')
        local_path = os.path.join(ROOT_DIR, 'data', 'input', playlist.title, video.title)
        download_subtitles(video.watch_url, local_path)
        local_path += '.tr.vtt' 





