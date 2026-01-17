from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from loguru import logger
from tqdm import tqdm

from src.settings import BaseS3Settings


class S3Downloader:
    def __init__(self, s3_settings: BaseS3Settings) -> None:
        """
        Initializes S3 downloader with connection settings.

        :param s3_settings: Configuration object with S3 credentials
        """
        self.bucket = s3_settings.s3_bucket
        self.base_prefix = s3_settings.s3_prefix.strip("/")

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=s3_settings.s3_key_id,
            aws_secret_access_key=s3_settings.s3_secret,
            endpoint_url=s3_settings.s3_url,
        )
        logger.info(f"S3 downloader initialized for bucket: {self.bucket}")

    def download_file(
        self,
        s3_path: Path | str,
        local_dir: Path,
    ) -> None:
        """
        Downloads single file from S3 to local directory.

        :param s3_path: Full S3 path to the file
        :param local_dir: Local directory to save the file
        """
        s3_path_str, full_key = self._normalize_path(s3_path)
        filename = Path(s3_path_str).name
        local_path = local_dir / filename

        logger.info(f"Downloading file: {full_key}")
        local_dir.mkdir(parents=True, exist_ok=True)
        self._download_file(full_key, local_path)
        logger.info(f"File downloaded: {local_path}")

    def download_directory(self, s3_prefix: str | Path, local_dir: Path) -> None:
        """
        Downloads entire directory from S3 to local path.

        :param s3_prefix: S3 prefix (directory path) to download
        :param local_dir: Local directory to save files
        """
        _, full_prefix = self._normalize_path(s3_prefix)

        logger.info(f"Downloading S3 directory: {full_prefix}")
        local_dir.mkdir(parents=True, exist_ok=True)

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=full_prefix)

        downloaded_files_count = 0
        page_n = 0
        for page in pages:
            page_n += 1
            for obj in tqdm(page.get("Contents", []), desc=f"Download files (page {page_n})"):
                if obj["Key"].endswith("/"):
                    continue

                rel_path = Path(obj["Key"]).relative_to(full_prefix)
                local_path = local_dir / rel_path

                local_path.parent.mkdir(parents=True, exist_ok=True)
                self._download_file(obj["Key"], local_path)
                downloaded_files_count += 1

        logger.info(f"Directory downloaded to {local_dir} ({downloaded_files_count} files)")

    def list_files_in_folder(self, additional_prefix: str = "", verbose: bool = True) -> None:
        prefix = str(Path(self.base_prefix) / additional_prefix)
        total_files = 0
        continuation_token = None
        first_page = True

        logger.info(f"Listing files in folder: '{prefix}'")

        while True:
            request_kwargs = {
                "Bucket": self.bucket,
                "Prefix": prefix,
            }
            if continuation_token:
                request_kwargs["ContinuationToken"] = continuation_token

            response = self.s3_client.list_objects_v2(**request_kwargs)

            if "Contents" in response:
                total_files += len(response["Contents"])

                if verbose and first_page:
                    for obj in response["Contents"]:
                        logger.info(obj["Key"])
                    first_page = False

            if response.get("NextContinuationToken"):
                continuation_token = response["NextContinuationToken"]
            else:
                break

        if total_files > 0:
            logger.info(f"Total files in '{prefix}': {total_files}")
        else:
            logger.error(f"No files found in '{prefix}'")

    def _normalize_path(self, s3_path: str | Path) -> tuple[str, str]:
        s3_path_str = str(s3_path).replace("\\", "/").strip("/")
        full_key = f"{self.base_prefix}/{s3_path_str}" if self.base_prefix else s3_path_str
        return s3_path_str, full_key

    def _download_file(self, full_key: str, local_path: Path) -> None:
        try:
            self.s3_client.download_file(Bucket=self.bucket, Key=full_key, Filename=str(local_path))
        except ClientError as e:
            logger.error(f"Download failed: {full_key} - {e}")
            raise RuntimeError(f"S3 download error: {e}") from e
