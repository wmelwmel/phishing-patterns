import shutil
from pathlib import Path

import py7zr
import pyzipper
from loguru import logger

from src.settings import ArchiveExtractorSettings


class ArchiveExtractor:
    def __init__(
        self,
        extract_dir: Path,
    ):
        """
        Initializes the archive extractor.

        :param data_dir: Directory containing archive files
        :param extract_dir_name: Name for the extraction subdirectory
        """
        self.extract_base_dir = extract_dir
        extractor_settings = ArchiveExtractorSettings()
        self.archive_passwords = {".zip": extractor_settings.mails_pass, ".7z": extractor_settings.infected_pass}

        logger.info(f"Initialized extractor. Data extract directory: {self.extract_base_dir}")

    def extract_archive(self, archive_path: str | Path, reextract: bool = True) -> Path:
        """
        Extracts an archive to dedicated directory.

        :param archive_path: Path to archive file
        :param reextract: Whether to overwrite existing extraction
        :return: Path to extracted directory
        """
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            error_msg = f"Archive not found: {archive_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        extract_path = self._get_archive_extract_path(archive_path.stem)

        if extract_path.exists():
            if reextract:
                logger.warning(f"Removing existing directory: {extract_path}")
                shutil.rmtree(extract_path)
            else:
                logger.info(f"Directory exists. Skipping extraction: {extract_path}")
                return extract_path

        extract_path.mkdir(parents=True, exist_ok=True)

        try:
            suffix = archive_path.suffix.lower()
            password = self._get_archive_password(suffix)
            if suffix == ".7z":
                self._extract_7z(archive_path, extract_path, password)
            elif suffix == ".zip":
                self._extract_zip(archive_path, extract_path, password)
            else:
                error_msg = f"Unsupported archive format: {suffix}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as error:
            error_msg = f"Extraction error: {error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Extracted successfully {archive_path} to {extract_path}")
        return extract_path

    def extract_all(
        self,
        archives_dir: Path,
        reextract: bool = False,
    ) -> None:
        """
        Extracts all supported archives in data directory.

        :param archives_dir: Path with archives
        :param reextract: Whether to overwrite existing extractions
        """
        logger.info(f"Processing archives from {archives_dir}")

        results = {}

        for archive_path in archives_dir.rglob("*"):
            try:
                extract_path = self.extract_archive(archive_path, reextract)
                results[archive_path.name] = extract_path
            except Exception as error:
                logger.error(f"Failed processing {archive_path.name}: {error}")

        logger.info(f"Processed {len(results)} archives")

    def _get_archive_extract_path(self, archive_stem: str) -> Path:
        return self.extract_base_dir / archive_stem

    def _get_archive_password(self, archive_extension: str) -> str | None:
        return self.archive_passwords.get(archive_extension)

    def _extract_7z(self, archive_path: Path, extract_path: Path, password: str | None) -> None:
        with py7zr.SevenZipFile(archive_path, mode="r", password=password) as archive:
            archive.extractall(extract_path)

    def _extract_zip(self, archive_path: Path, extract_path: Path, password: str | None) -> None:
        with pyzipper.AESZipFile(archive_path, "r") as archive:
            if password:
                archive.setpassword(password.encode("utf-8"))
            archive.extractall(extract_path)
