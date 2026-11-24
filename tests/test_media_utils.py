import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from media_utils import get_youtube_video_id


class MediaUtilsTests(unittest.TestCase):
    def test_extracts_supported_youtube_url_forms(self):
        examples = {
            'https://youtu.be/dQw4w9WgXcQ?t=12': 'dQw4w9WgXcQ',
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=playlist': 'dQw4w9WgXcQ',
            'http://m.youtube.com/shorts/dQw4w9WgXcQ': 'dQw4w9WgXcQ',
            'https://music.youtube.com/embed/dQw4w9WgXcQ': 'dQw4w9WgXcQ',
        }
        for url, expected_id in examples.items():
            with self.subTest(url=url):
                self.assertEqual(get_youtube_video_id(url), expected_id)

    def test_rejects_non_youtube_and_malformed_urls(self):
        for url in (
            '',
            'file:///tmp/song.mp3',
            'https://example.com/watch?v=abc123',
            'https://youtube.com/watch',
            'https://[broken/watch?v=abc123',
            'https://youtu.be/../',
        ):
            with self.subTest(url=url):
                self.assertIsNone(get_youtube_video_id(url))

    def test_playlist_id_is_used_only_when_requested(self):
        url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=playlist'
        self.assertEqual(get_youtube_video_id(url), 'dQw4w9WgXcQ')
        self.assertEqual(get_youtube_video_id(url, ignore_playlist=False), 'playlist')


if __name__ == '__main__':
    unittest.main()
