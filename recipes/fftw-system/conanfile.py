from conan import ConanFile
from conan.tools.system.package_manager import Apt, Brew, Dnf


class SystemFftw(ConanFile):
    name = "fftw-system"
    description = "System FFTW double- and single-precision runtimes"
    package_type = "shared-library"
    settings = "os", "arch"
    options = {"development": [True, False]}
    default_options = {"development": False}

    def system_requirements(self):
        apt_packages = ["libfftw3-dev"] if self.options.development else [
            "libfftw3-double3", "libfftw3-single3"
        ]
        rpm_packages = ["fftw-devel"] if self.options.development else [
            "fftw-libs-double", "fftw-libs-single"
        ]
        Apt(self).install(apt_packages, check=True)
        Brew(self).install(["fftw"], check=True)
        Dnf(self).install(rpm_packages, check=True)

    def package_info(self):
        # syclfft's FindFFTW3 verifies both installed precisions against this version.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        self.cpp_info.includedirs = []
