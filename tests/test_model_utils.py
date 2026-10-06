import os
import socket
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import model_utils


class ModelUtilsTests(unittest.TestCase):
    def test_validates_names_for_windows_and_unix(self):
        self.assertEqual(model_utils.validate_model_name('My Voice (v2)'), 'My Voice (v2)')
        for name in ('', '..', '../outside', 'folder\\voice', 'CON', 'name.', '.hidden'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                model_utils.validate_model_name(name)

    def test_derives_names_from_encoded_and_pixeldrain_urls(self):
        self.assertEqual(
            model_utils.model_name_from_url(
                'https://huggingface.co/user/repo/resolve/main/My%20Voice.zip?download=true'
            ),
            'My Voice',
        )
        self.assertEqual(
            model_utils.model_name_from_url('https://pixeldrain.com/u/abc123'),
            'abc123',
        )

    def test_installs_only_the_model_and_optional_index(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            archive_path = root_path / 'voice.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('nested/Voice.pth', b'checkpoint')
                archive.writestr('nested/Voice.index', b'index')
                archive.writestr('README.txt', b'unused')

            installed = Path(
                model_utils.install_model_archive(archive_path, root_path / 'models', 'Voice')
            )
            self.assertEqual((installed / 'Voice.pth').read_bytes(), b'checkpoint')
            self.assertEqual((installed / 'Voice.index').read_bytes(), b'index')
            self.assertEqual({path.name for path in installed.iterdir()}, {'Voice.pth', 'Voice.index'})

    def test_rejects_traversal_and_ambiguous_model_archives(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            traversal_archive = root_path / 'traversal.zip'
            with zipfile.ZipFile(traversal_archive, 'w') as archive:
                archive.writestr('../outside.pth', b'checkpoint')
            models_dir = root_path / 'models'
            with self.assertRaisesRegex(ValueError, 'unsafe file path'):
                model_utils.install_model_archive(traversal_archive, models_dir, 'Traversal')
            self.assertFalse((root_path / 'outside.pth').exists())

            ambiguous_archive = root_path / 'ambiguous.zip'
            with zipfile.ZipFile(ambiguous_archive, 'w') as archive:
                archive.writestr('first.pth', b'first')
                archive.writestr('second.pth', b'second')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                model_utils.install_model_archive(ambiguous_archive, models_dir, 'Ambiguous')

    @patch(
        'model_utils.socket.getaddrinfo',
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 80))],
    )
    def test_rejects_private_download_hosts(self, resolve_host):
        with self.assertRaisesRegex(ValueError, 'public IP addresses'):
            model_utils._validate_public_http_url('http://example.test/model.zip')

    @patch('model_utils.socket.getaddrinfo')
    @patch('model_utils.requests.Session')
    def test_downloads_from_public_hosts_with_a_bounded_stream(
        self, session_factory, resolve_host
    ):
        resolve_host.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))
        ]
        response = MagicMock()
        response.is_redirect = False
        response.headers = {'Content-Length': str(1024 * 1024 + 1)}
        response.iter_content.return_value = [b'x' * (1024 * 1024), b'x']
        session = session_factory.return_value
        session.get.return_value.__enter__.return_value = response
        session.get.return_value.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, 'model.zip')
            with patch.dict(os.environ, {'AICOVERGEN_MAX_MODEL_DOWNLOAD_MB': '1'}):
                with self.assertRaisesRegex(ValueError, 'configured download size limit'):
                    model_utils.download_model_archive(
                        'https://example.test/model.zip', destination
                    )
            self.assertFalse(os.path.exists(destination))
        session.close.assert_called_once()

    @patch('model_utils.socket.getaddrinfo')
    @patch('model_utils.requests.Session')
    def test_rejects_incomplete_downloads(self, session_factory, resolve_host):
        resolve_host.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))
        ]
        response = MagicMock()
        response.is_redirect = False
        response.headers = {'Content-Length': '6'}
        response.iter_content.return_value = [b'model']
        session = session_factory.return_value
        session.get.return_value.__enter__.return_value = response
        session.get.return_value.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, 'model.zip')
            with self.assertRaisesRegex(ValueError, 'download was incomplete'):
                model_utils.download_model_archive(
                    'https://example.test/model.zip', destination
                )
            self.assertFalse(os.path.exists(destination))

    @patch('model_utils.socket.getaddrinfo')
    @patch('model_utils.requests.Session')
    def test_saves_complete_downloads(self, session_factory, resolve_host):
        resolve_host.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))
        ]
        response = MagicMock()
        response.is_redirect = False
        response.headers = {'Content-Length': '5'}
        response.iter_content.return_value = [b'mo', b'del']
        session = session_factory.return_value
        session.get.return_value.__enter__.return_value = response
        session.get.return_value.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, 'model.zip')
            result = model_utils.download_model_archive(
                'https://example.test/model.zip', destination
            )
            self.assertEqual(result, destination)
            with open(destination, 'rb') as archive_file:
                self.assertEqual(archive_file.read(), b'model')

    @patch('model_utils.socket.getaddrinfo')
    @patch('model_utils.requests.Session')
    def test_validates_redirect_destinations(self, session_factory, resolve_host):
        resolve_host.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))
        ]
        response = MagicMock()
        response.is_redirect = True
        response.headers = {'Location': 'http://127.0.0.1/internal'}
        session = session_factory.return_value
        session.get.return_value.__enter__.return_value = response
        session.get.return_value.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, 'model.zip')
            with self.assertRaisesRegex(ValueError, 'public IP addresses'):
                model_utils.download_model_archive(
                    'https://example.test/model.zip', destination
                )
            session.get.assert_called_once()
            self.assertFalse(os.path.exists(destination))
        session.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
