def protect(videos: dict) -> dict:
    """Return videos unchanged. Use to verify the pipeline runs."""
    return {n: v.copy() for n, v in videos.items()}
