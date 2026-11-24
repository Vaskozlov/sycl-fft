import re

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
        if self.settings.os != "Linux" or not re.fullmatch(r"(?:1[5-9]|2[01])\.\d+\.\d+", str(self.version)):
            raise ConanInvalidConfiguration("Select an exact Linux Clang release from 15 through 21")

    def system_requirements(self):
        major = Version(self.version).major
        apt = Apt(self)
        if self.options.development:
            apt.install([f"clang-{major}", f"llvm-{major}-dev", f"libclang-{major}-dev"],
                        check=True)
        else:
            apt.install_substitutes([f"libclang-cpp{major}"], [f"libclang-cpp{major}t64"],
                                    check=True)

    def package_info(self):
        # LLVM and Clang provide their own versioned CMake packages.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.includedirs = []
