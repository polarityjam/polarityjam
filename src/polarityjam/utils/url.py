"""Contains url related functions."""
import re
from enum import Enum, unique
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from polarityjam.polarityjam_logging import get_logger
from polarityjam.utils.io import create_path_recursively


@unique
class ResponseStatus(Enum):
    """Response values and their name."""

    OK = 200  # response included
    Created = 201  # response included
    Accepted = 202  # response included
    NoContent = 204  # response NOT included
    BadRequest = 400  # error response included
    Unauthorized = 401  # error response included
    Forbidden = 403  # error response included
    NotFound = 404  # error response included
    MethodNotAllowed = 405  # error response included
    Conflict = 409  # error response included
    UnsupportedMediaType = 415  # error response included
    TooManyRequests = 429  # error response included
    InternalServerError = 500  # error NOT response included


def is_downloadable(url):
    """Show if url is a downloadable resource."""
    with _get_session() as s:
        h = s.head(url, allow_redirects=True)
        header = h.headers
        content_type = header.get("content-type")
        if "html" in content_type.lower():
            return False
        return True


def download_resource(url, path):
    """Download a resource given its url."""
    get_logger().debug(f"Download url {url} to {path}...")

    path = Path(path)

    if not is_downloadable(url):
        raise AssertionError('Resource "%s" not downloadable!' % url)

    r = _request_get(url)

    create_path_recursively(path.parent)
    with open(path, "wb") as f:
        for chunk in r:
            f.write(chunk)

    return path


def _get_session():
    s = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)

    adapter = HTTPAdapter(max_retries=retry)

    s.mount("http://", adapter)
    s.mount("https://", adapter)

    return s


def _request_get(url):
    """Get a response from a request to a resource url."""
    with _get_session() as s:
        r = s.get(url, allow_redirects=True, stream=True)

        if r.status_code != ResponseStatus.OK.value:
            raise ConnectionError("Could not connect to resource %s!" % url)

        return r


def retrieve_redirect_url(url):
    """Retrieve the redirect url of a resource."""
    with _get_session() as s:
        r = s.get(url, allow_redirects=True, stream=False)

        if r.status_code != ResponseStatus.OK.value:
            raise ConnectionError("Could not connect to resource %s!" % url)

        return r.url


def is_url(str_input: str):
    """Pars a url."""
    url_regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(url_regex, str_input) is not None


def is_git_ssh_address(str_input: str):
    """Pars an ssh address."""
    git_regex = re.compile(
        r"(ssh://){0,1}"  # long ssh address start
        r"[\S]*@"  # user@
        r"[\S]*",  # host and project
        re.IGNORECASE,
    )
    return re.match(git_regex, str_input) is not None
