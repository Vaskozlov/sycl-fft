from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.scm import Version
from conan.tools.system.package_manager import Apt


class SystemClangCpp(ConanFile):
    name = "clang-cpp-system"
    description = "System Clang C++ frontend used by the syclfft PTX compiler"
    package_type = "shared-library"
    settings = "os", "arch"
    options = {"development": [True, False]}
    default_options = {"development": False}

    def validate(self):
        if self.settings.os != "Linux" or Version(self.version).major != "20":
            raise ConanInvalidConfiguration("The VkFFT provider supports Linux with Clang 20")

    def system_requirements(self):
        major = Version(self.version).major
        packages = ([f"llvm-{major}-dev", f"libclang-{major}-dev"]
                    if self.options.development else [f"libclang-cpp{major}"])
        Apt(self).install(packages, check=True)

    def package_info(self):
        # LLVM and Clang provide their own versioned CMake packages.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.includedirs = []
