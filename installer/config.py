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
    # A profile with a port runs in its own always-on llama-server on that
    # port instead of the router, so the router never unloads it.
    port: int | None = None

    @property
    def resident(self) -> bool:
        return self.port is not None


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
        # Multi-file: the include pattern is either a shard directory
        # (Q4_K_M/*) or root-level shards (Name-Q4_K_M-*.gguf). Glob it
        # relative to the model dir and take the first shard.
        shards = sorted(
            p for p in Path(self.local_dir).glob(self.include or "")
            if p.suffix == ".gguf"
        )
        if shards:
            return str(shards[0])
        prefix = self.include.split("/")[0] if self.include and "/" in self.include else ""
        return f"{self.local_dir}/{prefix}/" if prefix else f"{self.local_dir}/"


@dataclass
class Service:
    """A Python network service run from its own venv (e.g. Wyoming STT/TTS)."""
    name: str
    package: str
    port: int
    module: str | None = None
    script: str | None = None
    args: list[str] = field(default_factory=list)
    env_file: str | None = None
    # Hugging Face repo downloaded into the data dir before the service starts.
    download: str | None = None

    @property
    def root(self) -> str:
        return f"/srv/llm/services/{self.name}"

    @property
    def data_dir(self) -> str:
        return f"{self.root}/data"

    @property
    def download_dir(self) -> str | None:
        if not self.download:
            return None
        return f"{self.data_dir}/{self.download.split('/')[-1]}"


@dataclass
class Defaults:
    load_mode: str = "auto"
    parallel: int = 1
    cache_type_k: str = "q4_0"
    cache_type_v: str = "q4_0"
    device: str = "Vulkan0"
    gpu_layers: str = "auto"
    ctx_size: int = 8192
    no_warmup: bool = True
    cache_ram: int = 8192


@dataclass
class Config:
    models_toml: Path
    models: list[Model]
    defaults: Defaults
    services: list[Service] = field(default_factory=list)
    llama_cpp_ref: str = "latest"
    models_max: int = 1
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
        load_mode=defaults_raw.get("load_mode", "auto"),
        parallel=defaults_raw.get("parallel", 1),
        cache_type_k=defaults_raw.get("cache_type_k", "q4_0"),
        cache_type_v=defaults_raw.get("cache_type_v", "q4_0"),
        device=defaults_raw.get("device", "Vulkan0"),
        gpu_layers=defaults_raw.get("gpu_layers", "auto"),
        ctx_size=defaults_raw.get("ctx_size", 8192),
        no_warmup=defaults_raw.get("no_warmup", True),
        cache_ram=defaults_raw.get("cache_ram", 8192),
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
                port=profile_raw.get("port"),
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

    services = []
    for name, service_raw in raw.get("services", {}).items():
        if bool(service_raw.get("module")) == bool(service_raw.get("script")):
            print(f"ERROR: service {name}: set exactly one of 'module' or 'script'", file=sys.stderr)
            sys.exit(1)
        services.append(Service(
            name=name,
            package=service_raw["package"],
            port=service_raw["port"],
            module=service_raw.get("module"),
            script=service_raw.get("script"),
            args=service_raw.get("args", []),
            env_file=service_raw.get("env_file"),
            download=service_raw.get("download"),
        ))

    # The router, resident profiles and services all bind 0.0.0.0.
    ports = [args.port]
    ports += [p.port for m in models for p in m.profiles if p.resident]
    ports += [s.port for s in services]
    if len(ports) != len(set(ports)):
        print(f"ERROR: port collision between router, resident profiles and services: {ports}",
              file=sys.stderr)
        sys.exit(1)

    build_raw = raw.get("build", {})
    llama_cpp_ref = build_raw.get("llama_cpp_ref", "latest")
    models_max = raw.get("router", {}).get("models_max", 1)

    return Config(
        models_toml=toml_path,
        models=models,
        defaults=defaults,
        services=services,
        llama_cpp_ref=llama_cpp_ref,
        models_max=models_max,
        prune=args.prune,
        dry_run=args.dry_run,
        host=args.host,
        port=args.port,
    )
