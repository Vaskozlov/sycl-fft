from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.scm import Version
from conan.tools.system.package_manager import Apt


class SystemCudaToolkit(ConanFile):
    name = "cuda-toolkit-system"
    description = "NVIDIA CUDA Runtime and cuFFT used by the optional cuFFT provider"
    package_type = "shared-library"
    settings = "os", "arch"
    options = {"development": [True, False]}
    default_options = {"development": False}

    def validate(self):
        if self.settings.os not in ("Linux", "Windows"):
            raise ConanInvalidConfiguration("The cuFFT provider requires Linux or Windows")

    def system_requirements(self):
        # NVIDIA's Debian packages use the toolkit major/minor in their names.
        release = f"{Version(self.version).major}-{Version(self.version).minor}"
        suffix = f"dev-{release}" if self.options.development else release
        Apt(self).install([f"cuda-cudart-{suffix}", f"libcufft-{suffix}"], check=True)

    def package_info(self):
        # FindCUDAToolkit supplies the imported CUDA targets for the pinned release.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.includedirs = []
