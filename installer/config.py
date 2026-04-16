"""Load and validate models.toml."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    # Python 3.10 fallback — unlikely on Bookworm but costs nothing.
    try:
        import tomllib  # type: ignore[import]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import,no-redef]


@dataclass
class Profile:
    name: str
    ctx_size: int = 8192
    gpu_layers: str | int = "auto"
    device: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Model:
    id: str
    repo: str
    file: str | None = None
    include: str | None = None
    large: bool = False
    dir: str | None = None
    profiles: list[Profile] = field(default_factory=list)

    @property
    def local_dir(self) -> str:
        return f"/srv/llm/models/{self.dir or self.id}"

    @property
    def first_gguf(self) -> str:
        """Return the path to the primary GGUF for the models.ini model= line.

        For multi-file models, scans disk for the first shard (sorted).
        Falls back to a directory path if files aren't downloaded yet.
        """
        if self.file:
            return f"{self.local_dir}/{self.file}"
        # Multi-file (include pattern like Q4_K_M/*): find first shard on disk.
        prefix = self.include.split("/")[0] if self.include else ""
        shard_dir = Path(self.local_dir) / prefix
        if shard_dir.is_dir():
            shards = sorted(shard_dir.glob("*.gguf"))
            if shards:
                return str(shards[0])
        return f"{self.local_dir}/{prefix}/"


@dataclass
class Defaults:
    mmap: bool = True
    parallel: int = 1
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"
    device: str | None = None
    gpu_layers: str = "auto"
    ctx_size: int = 8192
    no_warmup: bool = True


@dataclass
class Config:
    models_toml: Path
    models: list[Model]
    defaults: Defaults
    llama_cpp_ref: str = "latest"
    prune: bool = False
    dry_run: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    repo_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)


def load_config(args) -> Config:
    if args.models_toml:
        toml_path = Path(args.models_toml)
    else:
        toml_path = Path(__file__).resolve().parent.parent / "models.toml"

    if not toml_path.exists():
        print(f"ERROR: {toml_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    defaults_raw = raw.get("defaults", {})
    defaults = Defaults(
        mmap=defaults_raw.get("mmap", True),
        parallel=defaults_raw.get("parallel", 1),
        cache_type_k=defaults_raw.get("cache_type_k", "q4_0"),
        cache_type_v=defaults_raw.get("cache_type_v", "q4_0"),
        device=defaults_raw.get("device"),
        gpu_layers=defaults_raw.get("gpu_layers", "auto"),
        ctx_size=defaults_raw.get("ctx_size", 8192),
        no_warmup=defaults_raw.get("no_warmup", True),
    )

    models = []
    for model_id, model_raw in raw.get("models", {}).items():
        profiles = []
        for _profile_key, profile_raw in model_raw.get("profiles", {}).items():
            profiles.append(Profile(
                name=profile_raw["name"],
                ctx_size=profile_raw.get("ctx_size", defaults.ctx_size),
                gpu_layers=profile_raw.get("gpu_layers", defaults.gpu_layers),
                device=profile_raw.get("device"),
                extra=profile_raw.get("extra", {}),
            ))
        models.append(Model(
            id=model_id,
            repo=model_raw["repo"],
            file=model_raw.get("file"),
            include=model_raw.get("include"),
            large=model_raw.get("large", False),
            dir=model_raw.get("dir"),
            profiles=profiles,
        ))

    build_raw = raw.get("build", {})
    llama_cpp_ref = build_raw.get("llama_cpp_ref", "latest")

    return Config(
        models_toml=toml_path,
        models=models,
        defaults=defaults,
        llama_cpp_ref=llama_cpp_ref,
        prune=args.prune,
        dry_run=args.dry_run,
        host=args.host,
        port=args.port,
    )
