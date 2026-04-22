from huggingface_hub import hf_hub_download, snapshot_download


def resolve_hf_file(repo_id, filename, cache_dir=None):
    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        return hf_hub_download(local_files_only=True, **kwargs)
    except Exception:
        return hf_hub_download(local_files_only=False, **kwargs)


def resolve_hf_snapshot(repo_id, cache_dir=None):
    kwargs = {
        "repo_id": repo_id,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        return snapshot_download(local_files_only=True, **kwargs)
    except Exception:
        return snapshot_download(local_files_only=False, **kwargs)


def resolve_hf_download_kwargs(repo_id, filename, cache_dir=None):
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=True,
            **kwargs,
        )
        kwargs["local_files_only"] = True
    except Exception:
        kwargs["local_files_only"] = False

    return kwargs
