import boto3
import os
from typing import Optional


def create_s3_client(
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> boto3.client:
    """Create an S3 client configured for S3-compatible storage services.

    This function creates a boto3 S3 client with optimized settings for reliable
    data transfer. It supports both direct credential passing and environment
    variable configuration.

    Args:
        access_key_id: S3 access key ID. If None, reads from AWS_ACCESS_KEY_ID env var
        secret_access_key: S3 secret key. If None, reads from AWS_SECRET_ACCESS_KEY env var
        endpoint_url: S3-compatible storage service endpoint URL

    Returns:
        boto3.client: Configured S3 client with optimized settings

    Environment Variables:
        - AWS_ACCESS_KEY_ID: S3 access key ID (if not provided as argument)
        - AWS_SECRET_ACCESS_KEY: S3 secret key (if not provided as argument)

    Example:
        ```python
        # Using environment variables
        s3_client = create_s3_client()

        # Using explicit credentials
        s3_client = create_s3_client(
            access_key_id="your_key",
            secret_access_key="your_secret",
            endpoint_url="your_endpoint_url"
        )
        ```

    Note:
        The client is configured with path-style addressing and S3v4 signatures
        for maximum compatibility with S3-compatible storage services.
    """
    access_key_id = access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_access_key = secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL")

    if not access_key_id or not secret_access_key:
        raise ValueError(
            "S3 credentials must be provided either through arguments or "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        )

    if not endpoint_url:
        raise ValueError(
            "S3 endpoint URL must be provided either through arguments or "
            "S3_ENDPOINT_URL environment variable"
        )

    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        endpoint_url=endpoint_url,
        use_ssl=True,
        verify=True,
        config=boto3.session.Config(
            s3={"addressing_style": "path"},
            signature_version="s3v4",
            # Advanced configuration options (currently commented out):
            # retries=dict(
            #     max_attempts=3,  # Number of retry attempts
            #     mode='adaptive'  # Adds exponential backoff
            # ),
            # max_pool_connections=20,  # Limits concurrent connections
            # connect_timeout=60,  # Connection timeout in seconds
            # read_timeout=300,    # Read timeout in seconds
            # tcp_keepalive=True,  # Enable TCP keepalive
        ),
    )
