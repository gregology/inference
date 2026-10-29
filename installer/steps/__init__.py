"""All installer steps, in execution order."""

from .packages import PackagesStep
from .user import UserStep
from .gpu_permissions import GpuPermissionsStep
from .vulkan_headers import VulkanHeadersStep
from .spirv_headers import SpirvHeadersStep
from .shaderc import ShadercStep
from .build_llama import BuildLlamaStep
from .huggingface import HuggingFaceStep
from .models import ModelsStep
from .router_config import RouterConfigStep
from .systemd import SystemdStep
from .resident import ResidentStep
from .services import ServicesStep

ALL_STEPS: list = [
    PackagesStep,
    UserStep,
    GpuPermissionsStep,
    VulkanHeadersStep,
    SpirvHeadersStep,
    ShadercStep,
    BuildLlamaStep,
    HuggingFaceStep,
    ModelsStep,
    RouterConfigStep,
    SystemdStep,
    ResidentStep,
    ServicesStep,
]
