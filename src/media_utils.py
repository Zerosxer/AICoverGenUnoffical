import re
from urllib.parse import parse_qs, urlsplit


YOUTUBE_HOSTS = {
    'm.youtube.com',
    'music.youtube.com',
    'youtube.com',
    'www.youtube.com',
}
YOUTUBE_VIDEO_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{11}$')


def _validated_video_id(value):
    return value if value and YOUTUBE_VIDEO_ID_PATTERN.fullmatch(value) else None


def get_youtube_video_id(url, ignore_playlist=True):
    try:
        parsed_url = urlsplit((url or '').strip())
        hostname = (parsed_url.hostname or '').lower()
    except ValueError:
        return None

    if parsed_url.scheme.lower() not in {'http', 'https'}:
        return None

    if hostname == 'youtu.be':
        return _validated_video_id(parsed_url.path.strip('/').split('/')[0])

    if hostname not in YOUTUBE_HOSTS:
        return None

    if not ignore_playlist:
        playlist_id = parse_qs(parsed_url.query).get('list', [None])[0]
        if playlist_id:
            return playlist_id

    if parsed_url.path == '/watch':
        return _validated_video_id(parse_qs(parsed_url.query).get('v', [None])[0])

    path_parts = [part for part in parsed_url.path.split('/') if part]
    if len(path_parts) >= 2 and path_parts[0] in {'embed', 'live', 'shorts', 'v'}:
        return _validated_video_id(path_parts[1])

    return None
