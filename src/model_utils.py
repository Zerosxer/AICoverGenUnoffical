import ipaddress
import os
import re
import shutil
import socket
import stat
import tempfile
import zipfile
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlparse

import requests


def validate_model_name(name):
    model_name = (name or '').strip()
    windows_device_name = model_name.split('.')[0].upper()
    if (
        not model_name
        or model_name.startswith('.')
        or model_name in {'.', '..'}
        or model_name != os.path.basename(model_name)
        or model_name.rstrip(' .') != model_name
        or windows_device_name in {'CON', 'PRN', 'AUX', 'NUL'}
        or re.fullmatch(r'(COM|LPT)[1-9]', windows_device_name)
        or not re.fullmatch(r'[\w .()-]+', model_name)
    ):
        raise ValueError('Model name is empty or contains unsupported characters.')
    return model_name


def model_name_from_url(url):
    parsed_url = urlparse(url)
    filename = unquote(PurePosixPath(parsed_url.path).name)
    if filename.lower() in {'', 'u'} and 'pixeldrain.com' in (parsed_url.hostname or '').lower():
        filename = PurePosixPath(parsed_url.path.rstrip('/')).name
    name = os.path.splitext(filename)[0]
    if name.lower() in {'resolve', 'file'}:
        raise ValueError('Cannot determine a model name from this URL; provide a direct ZIP URL.')
    return validate_model_name(name)


def _limit_from_env(variable_name, default_mb):
    value = os.environ.get(variable_name)
    if value is None:
        return default_mb * 1024 * 1024
    try:
        megabytes = int(value)
    except ValueError as error:
        raise ValueError(f'{variable_name} must be a positive integer (MiB).') from error
    if megabytes <= 0:
        raise ValueError(f'{variable_name} must be a positive integer (MiB).')
    return megabytes * 1024 * 1024


def _validate_public_http_url(url):
    parsed_url = urlparse(url)
    if (
        parsed_url.scheme not in {'http', 'https'}
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError('Model download URL must be a public HTTP or HTTPS URL.')

    try:
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
    except ValueError as error:
        raise ValueError('Model download URL contains an invalid port.') from error

    try:
        address = ipaddress.ip_address(parsed_url.hostname)
        addresses = [address]
    except ValueError:
        try:
            addresses = [
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(
                    parsed_url.hostname, port, type=socket.SOCK_STREAM
                )
            ]
        except OSError as error:
            raise ValueError('Could not resolve model download host.') from error

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError('Model download URL must resolve only to public IP addresses.')
    return parsed_url


def download_model_archive(url, destination, progress=None):
    max_bytes = _limit_from_env('AICOVERGEN_MAX_MODEL_DOWNLOAD_MB', 4096)
    current_url = url
    parsed_url = urlparse(current_url)
    if (parsed_url.hostname or '').lower() in {'pixeldrain.com', 'www.pixeldrain.com'}:
        parts = [part for part in parsed_url.path.split('/') if part]
        if len(parts) == 2 and parts[0] == 'u':
            current_url = f'https://pixeldrain.com/api/file/{parts[1]}'
    session = requests.Session()
    downloaded = 0

    try:
        for _ in range(10):
            _validate_public_http_url(current_url)
            with session.get(
                current_url,
                stream=True,
                allow_redirects=False,
                timeout=(15, 120),
            ) as response:
                if response.is_redirect:
                    location = response.headers.get('Location')
                    if not location:
                        raise ValueError('Model download redirect did not include a destination.')
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_length = response.headers.get('Content-Length')
                expected_size = int(content_length) if content_length else None
                if expected_size is not None and expected_size > max_bytes:
                    raise ValueError('Model archive exceeds the configured download size limit.')

                with open(destination, 'wb') as archive_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError('Model archive exceeds the configured download size limit.')
                        archive_file.write(chunk)
                        if progress:
                            progress(downloaded, max_bytes)
                if expected_size is not None and downloaded != expected_size:
                    raise ValueError(
                        f'Model archive download was incomplete ({downloaded} of {expected_size} bytes).'
                    )
            return destination
        raise ValueError('Model download exceeded the redirect limit.')
    except Exception:
        if os.path.exists(destination):
            os.remove(destination)
        raise
    finally:
        session.close()


def install_model_archive(archive_path, models_dir, model_name):
    model_name = validate_model_name(model_name)
    models_root = os.path.realpath(models_dir)
    os.makedirs(models_root, exist_ok=True)
    target_dir = os.path.realpath(os.path.join(models_root, model_name))
    if os.path.commonpath([models_root, target_dir]) != models_root:
        raise ValueError('Model destination must stay inside the model directory.')
    if os.path.exists(target_dir):
        raise FileExistsError(f'Model directory already exists: {model_name}')

    max_unpacked_bytes = _limit_from_env('AICOVERGEN_MAX_MODEL_UNPACKED_MB', 8192)
    max_entries = 4096
    staging_dir = tempfile.mkdtemp(prefix='.model-install-', dir=models_root)
    try:
        with zipfile.ZipFile(archive_path, 'r') as archive:
            members = archive.infolist()
            if len(members) > max_entries:
                raise ValueError('Model archive contains too many files.')

            total_size = 0
            model_members = []
            index_members = []
            for member in members:
                member_path = member.filename
                path = PurePosixPath(member_path)
                mode = member.external_attr >> 16
                if (
                    not member_path
                    or '\x00' in member_path
                    or '\\' in member_path
                    or path.is_absolute()
                    or any(part in {'.', '..'} for part in path.parts)
                    or any(':' in part for part in path.parts)
                    or (mode & 0o170000) == stat.S_IFLNK
                ):
                    raise ValueError('Model archive contains an unsafe file path or link.')

                total_size += member.file_size
                if total_size > max_unpacked_bytes:
                    raise ValueError('Model archive exceeds the configured extracted size limit.')
                if member.is_dir():
                    continue

                suffix = path.suffix.lower()
                if suffix == '.pth':
                    if member.file_size == 0:
                        raise ValueError('Model archive contains an empty .pth file.')
                    model_members.append(member)
                elif suffix == '.index':
                    if member.file_size == 0:
                        raise ValueError('Model archive contains an empty .index file.')
                    index_members.append(member)

            if len(model_members) != 1:
                raise ValueError(
                    f'Model archive must contain exactly one .pth file; found {len(model_members)}.'
                )
            if len(index_members) > 1:
                raise ValueError(
                    f'Model archive must contain at most one .index file; found {len(index_members)}.'
                )

            selected_members = model_members + index_members
            basenames = [PurePosixPath(member.filename).name.lower() for member in selected_members]
            if len(set(basenames)) != len(basenames):
                raise ValueError('Model and index files must have different file names.')

            for member in selected_members:
                destination = os.path.join(staging_dir, PurePosixPath(member.filename).name)
                with archive.open(member, 'r') as source, open(destination, 'wb') as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)

        os.replace(staging_dir, target_dir)
        staging_dir = None
        return target_dir
    finally:
        if staging_dir and os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
