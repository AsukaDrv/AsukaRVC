import os
import re
import wget
import shutil
import zipfile
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlencode, parse_qs, urlparse

from programs.rvc.lib.utils import format_title
from programs.rvc.lib.tools import gdown

# Repository root, resolved from this file so the paths below are independent
# of the process working directory. model_download.py lives at
# programs/rvc/lib/tools/model_download.py, so the project root is five
# directory levels up (tools -> lib -> rvc -> programs -> root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# NOTE: The old implementation located the storage root with
# find_folder_parent(os.getcwd(), "logs"), which walked the whole tree looking
# for any folder named "logs". On a fresh clone that folder does not exist, so
# it returned None and os.path.join(None, "zips") crashed this module at import
# time (taking down programs/core.py and the whole WebUI with it). When a clone
# DOES contain .git/, the lookup matched .git/logs and returned .git as the
# storage root, writing downloads into .git/zips. Anchoring to PROJECT_ROOT
# avoids both failure modes.
file_path = os.path.join(PROJECT_ROOT, "logs")
zips_path = os.path.join(file_path, "zips")

# Where downloaded voice models are installed. The WebUI lists models from
# rvc_models/, so extracted models must land there; the previous target
# (logs/<name>) collided with the training experiments root and the downloaded
# models never appeared in the voice models dropdown.
rvc_models_path = os.path.join(PROJECT_ROOT, "rvc_models")

# All downloads and extractions below use absolute paths. The previous code
# called os.chdir() back and forth, which mutates the working directory of the
# ENTIRE process (all Gradio worker threads) and left the process inside
# zips_path whenever a download branch returned early.


def _filename_from_headers(response, url, default_name="download.zip"):
    """Best-effort file name from a download response.

    Falls back to the URL basename when the server does not send a usable
    Content-Disposition header (the old code raised KeyError/AttributeError
    there).
    """
    disposition = response.headers.get("Content-Disposition") or ""
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
    if match:
        return unquote(match.group(1).strip())
    url_name = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    return url_name or default_name


def _write_stream(response, destination_path, label="download"):
    """Stream a response body to destination_path with a console progress bar."""
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar_length = 50
    progress = 0
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            progress += len(data)
            if total_size_in_bytes:
                progress_percent = int((progress / total_size_in_bytes) * 100)
                num_dots = int((progress / total_size_in_bytes) * progress_bar_length)
                progress_bar = "[" + "." * num_dots + " " * (progress_bar_length - num_dots) + "]"
                print(f"{progress_percent}% {progress_bar} {progress}/{total_size_in_bytes}  ", end="\r")
                if progress_percent >= 100:
                    print("\n")
    if not total_size_in_bytes:
        print(f"{label} finished: {os.path.basename(destination_path)}")


def search_pth_index(folder):
    pth_paths = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file)) and file.endswith(".pth")
    ]
    index_paths = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file)) and file.endswith(".index")
    ]

    return pth_paths, index_paths


def get_mediafire_download_link(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    download_button = soup.find(
        "a", {"class": "input popsok", "aria-label": "Download file"}
    )
    if download_button:
        download_link = download_button.get("href")
        return download_link
    else:
        return None


def download_from_url(url):
    """Download a model archive into ``zips_path`` from the supported hosts.

    Returns "downloaded" on success, a host-specific status string, or None.
    Everything below uses absolute paths — never os.chdir(), which would
    mutate the working directory of the whole process (all threads).
    """
    os.makedirs(zips_path, exist_ok=True)
    if url != "":
        if "drive.google.com" in url:
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return None

            if file_id:
                try:
                    # Download straight into zips_path via the output= option
                    # instead of relying on the process working directory.
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}",
                        output=zips_path,
                        quiet=True,
                        fuzzy=True,
                    )
                except Exception as error:
                    error_message = str(error)
                    if (
                        "Too many users have viewed or downloaded this file recently"
                        in error_message
                    ):
                        return "too much use"
                    elif (
                        "Cannot retrieve the public link of the file." in error_message
                    ):
                        return "private link"
                    else:
                        print(f"An error occurred downloading the file: {error}")
                        return None
        elif "disk.yandex.ru" in url:
            base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
            public_key = url
            final_url = base_url + urlencode(dict(public_key=public_key))
            response = requests.get(final_url)
            download_url = response.json()["href"]
            download_response = requests.get(download_url)

            if download_response.status_code == 200:
                filename = parse_qs(urlparse(unquote(download_url)).query).get(
                    "filename", [""]
                )[0]
                if filename:
                    with open(os.path.join(zips_path, filename), "wb") as f:
                        f.write(download_response.content)
                else:
                    print("Failed to get filename from URL.")
                    return None
            else:
                print("Failed to get filename from URL.")
                return None

        elif "pixeldrain.com" in url:
            try:
                file_id = url.rstrip("/").split("pixeldrain.com/u/")[1]
                print(file_id)
                response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
                if response.status_code == 200:
                    file_name = _filename_from_headers(response, url)
                    with open(os.path.join(zips_path, file_name), "wb") as newfile:
                        newfile.write(response.content)
                    return "downloaded"
                else:
                    return None
            except Exception as error:
                print(f"An error occurred downloading the file: {error}")
                return None

        elif "cdn.discordapp.com" in url:
            file = requests.get(url)
            if file.status_code == 200:
                # The previous code wrote to a relative path after os.chdir()
                # and leaked the changed working directory on non-200 replies.
                file_name = unquote(url.rstrip("/").rsplit("/", 1)[-1])
                with open(os.path.join(zips_path, file_name), "wb") as newfile:
                    newfile.write(file.content)
            else:
                return None
        elif "/blob/" in url or "/resolve/" in url:
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/")

            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # The header may be missing (or lack a filename= entry) on some
                # hosts; fall back to the URL basename instead of crashing.
                file_name = _filename_from_headers(response, url)
                file_name = file_name.replace(os.path.sep, "_")
                _write_stream(response, os.path.join(zips_path, file_name))
            else:
                return None
        elif "/tree/main" in url:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            temp_url = ""
            for link in soup.find_all("a", href=True):
                if link["href"].endswith(".zip"):
                    temp_url = link["href"]
                    break
            if temp_url:
                # Normalize the found link to a direct-download URL and ALWAYS
                # download it: the old code placed wget.download() inside the
                # "not huggingface.co" branch, so plain HF /tree/ links were
                # resolved but never actually fetched.
                url = temp_url.replace("blob", "resolve")
                if "huggingface.co" not in url:
                    url = "https://huggingface.co" + url
                wget.download(url, out=zips_path)
            else:
                return None
        elif "applio.org" in url:
            parts = url.split("/")
            id_with_query = parts[-1]
            id_parts = id_with_query.split("?")
            id_number = id_parts[0]

            url = "https://cjtfqzjfdimgpvpwhzlv.supabase.co/rest/v1/models"
            headers = {
                "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNqdGZxempmZGltZ3B2cHdoemx2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTUxNjczODgsImV4cCI6MjAxMDc0MzM4OH0.7z5WMIbjR99c2Ooc0ma7B_FyGq10G8X-alkCYTkKR10"
            }

            params = {"id": f"eq.{id_number}"}
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                json_response = response.json()
                print(json_response)
                if json_response:
                    link = json_response[0]["link"]
                    verify = download_from_url(link)
                    if verify == "downloaded":
                        return "downloaded"
                    else:
                        return None
            else:
                return None
        else:
            try:
                wget.download(url, out=zips_path)
            except Exception as error:
                print(f"An error occurred downloading the file: {error}")
                return None

        # Normalize file names in the staging folder: only files directly in
        # zips_path, with absolute source and destination paths (the old code
        # built a relative destination that silently depended on os.chdir()).
        for currentPath, _, zipFiles in os.walk(zips_path):
            for file in zipFiles:
                filePart = file.split(".")
                extensionFile = filePart[-1]
                nameFile = "_".join(filePart[:-1])
                realPath = os.path.join(currentPath, file)
                targetPath = os.path.join(currentPath, nameFile + "." + extensionFile)
                if realPath != targetPath and not os.path.exists(targetPath):
                    os.rename(realPath, targetPath)

        return "downloaded"

    return None


def extract_and_show_progress(zipfile_path, unzips_path):
    try:
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                zip_ref.extract(file_info, unzips_path)
        os.remove(zipfile_path)
        return True
    except Exception as error:
        print(f"An error occurred extracting the zip file: {error}")
        return False


def unzip_file(zip_path, zip_file_name):
    zip_file_path = os.path.join(zip_path, zip_file_name + ".zip")
    extract_path = os.path.join(file_path, zip_file_name)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_file_path)


def model_download_pipeline(url: str):
    """Download a voice model from ``url`` and install it into rvc_models/.

    The WebUI lists voice models from ``rvc_models/<name>`` (see
    ui/tabs/common.py), so this pipeline installs there. The previous
    implementation extracted into ``logs/<name>`` — the training experiments
    root — where the model was invisible to the Inference tab.
    """
    try:
        verify = download_from_url(url)
        if verify == "downloaded":
            extract_folder_path = ""
            for filename in os.listdir(zips_path):
                if filename.endswith(".zip"):
                    zipfile_path = os.path.join(zips_path, filename)
                    print("Proceeding with the extraction...")

                    model_zip = os.path.basename(zipfile_path)
                    model_name = format_title(model_zip.split(".zip")[0])
                    extract_folder_path = os.path.join(
                        rvc_models_path,
                        os.path.normpath(model_name),
                    )
                    os.makedirs(extract_folder_path, exist_ok=True)
                    success = extract_and_show_progress(
                        zipfile_path, extract_folder_path
                    )

                    macosx_path = os.path.join(extract_folder_path, "__MACOSX")
                    if os.path.exists(macosx_path):
                        shutil.rmtree(macosx_path)

                    subfolders = [
                        f
                        for f in os.listdir(extract_folder_path)
                        if os.path.isdir(os.path.join(extract_folder_path, f))
                    ]
                    if len(subfolders) == 1:
                        subfolder_path = os.path.join(
                            extract_folder_path, subfolders[0]
                        )
                        for item in os.listdir(subfolder_path):
                            s = os.path.join(subfolder_path, item)
                            d = os.path.join(extract_folder_path, item)
                            shutil.move(s, d)
                        os.rmdir(subfolder_path)

                    for item in os.listdir(extract_folder_path):
                        if ".pth" in item:
                            file_name = item.split(".pth")[0]
                            if file_name != model_name:
                                os.rename(
                                    os.path.join(extract_folder_path, item),
                                    os.path.join(
                                        extract_folder_path, model_name + ".pth"
                                    ),
                                )
                        else:
                            if "v2" not in item:
                                if "_nprobe_1_" in item and "_v1" in item:
                                    file_name = item.split("_nprobe_1_")[1].split(
                                        "_v1"
                                    )[0]
                                    if file_name != model_name:
                                        new_file_name = (
                                            item.split("_nprobe_1_")[0]
                                            + "_nprobe_1_"
                                            + model_name
                                            + "_v1"
                                        )
                                        os.rename(
                                            os.path.join(extract_folder_path, item),
                                            os.path.join(
                                                extract_folder_path,
                                                new_file_name + ".index",
                                            ),
                                        )
                            else:
                                if "_nprobe_1_" in item and "_v2" in item:
                                    file_name = item.split("_nprobe_1_")[1].split(
                                        "_v2"
                                    )[0]
                                    if file_name != model_name:
                                        new_file_name = (
                                            item.split("_nprobe_1_")[0]
                                            + "_nprobe_1_"
                                            + model_name
                                            + "_v2"
                                        )
                                        os.rename(
                                            os.path.join(extract_folder_path, item),
                                            os.path.join(
                                                extract_folder_path,
                                                new_file_name + ".index",
                                            ),
                                        )

                    if success:
                        print(f"Model {model_name} downloaded!")
                    else:
                        print(f"Error downloading {model_name}")
                        return "Error"
            if extract_folder_path == "":
                print("Zip file was not found.")
                return "Error"
            result = search_pth_index(extract_folder_path)
            return result
        else:
            return "Error"
    except Exception as error:
        print(f"An unexpected error occurred: {error}")
        return "Error"
