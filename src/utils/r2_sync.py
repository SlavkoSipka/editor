"""Download R2 (S3-compatible) prefix into a local directory for Railway first boot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config


def sync_prefix(
    bucket: str,
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    prefix: str,
    dest: Path,
) -> int:
    prefix_clean = prefix.strip("/") + "/"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url.rstrip("/"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    dest.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_clean):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix_clean) :] if key.startswith(prefix_clean) else key
            local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local))
            n += 1
            if n % 1000 == 0:
                print(f"R2 sync … {n} files ({prefix_clean})", file=sys.stderr)
    print(f"R2 sync done: {n} files -> {dest} (prefix {prefix_clean})", file=sys.stderr)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True, help="Object prefix, e.g. library or embeddings")
    p.add_argument("--dest", required=True, type=Path, help="Local destination directory")
    args = p.parse_args()

    try:
        bucket = os.environ["R2_BUCKET"]
        endpoint = os.environ["R2_ENDPOINT_URL"]
        ak = os.environ["R2_ACCESS_KEY_ID"]
        sk = os.environ["R2_SECRET_ACCESS_KEY"]
    except KeyError as e:
        print(f"Missing env: {e.args[0]}", file=sys.stderr)
        sys.exit(1)

    sync_prefix(bucket, endpoint, ak, sk, args.prefix, args.dest)


if __name__ == "__main__":
    main()
