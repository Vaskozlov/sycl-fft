from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
import os


class SyclFftConan(ConanFile):
    name = "syclfft"
    version = "0.1.0"
    package_type = "shared-library"
    license = "Apache-2.0"
    url = "https://github.com/vaskozlov/sycl-fft"
    description = "FFTW-like reusable C++ FFT plans for SYCL USM"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "with_host": [True, False],
        "with_sycl": [True, False],
        "sycl_implementation": ["adaptivecpp", "dpcpp"],
        "use_system_fftw": [True, False],
        "with_cufft": ["auto", True, False],
    }
    default_options = {
        "with_host": True,
        "with_sycl": True,
        "sycl_implementation": "adaptivecpp",
        "use_system_fftw": False,
        "with_cufft": "auto",
        "fftw/*:shared": False,
        "fftw/*:precision_single": True,
        "fftw/*:precision_double": True,
        "fftw/*:precision_longdouble": False,
        "fftw/*:precision_quad": False,
        "fftw/*:openmp": False,
        "fftw/*:threads": False,
    }
    exports_sources = (
        "CMakeLists.txt", "cmake/*", "include/*", "src/*", "third_party/*",
        "examples/*", "tests/*", "LICENSE", "NOTICE", "README.md",
    )

    def config_options(self):
        if self.settings.os == "Macos":
            self.options.with_sycl = False
        elif self.settings.os == "Windows":
            self.options.sycl_implementation = "dpcpp"

    def configure(self):
        if not self.options.with_host:
            self.options.rm_safe("use_system_fftw")
        if not self.options.with_sycl:
            self.options.rm_safe("sycl_implementation")
            self.options.rm_safe("with_cufft")

    def requirements(self):
        if self.options.with_host and not self.options.get_safe("use_system_fftw", False):
            self.requires("fftw/3.3.10")

    def validate(self):
        if not self.options.with_host and not self.options.with_sycl:
            raise ConanInvalidConfiguration("At least one of with_host or with_sycl must be enabled")
        check_min_cppstd(self, "17")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["BUILD_SHARED_LIBS"] = True
        toolchain.cache_variables["SYCLFFT_BUILD_HOST"] = bool(self.options.with_host)
        toolchain.cache_variables["SYCLFFT_BUILD_SYCL"] = bool(self.options.with_sycl)
        toolchain.cache_variables["SYCLFFT_USE_SYSTEM_FFTW"] = bool(
            self.options.get_safe("use_system_fftw", False)
        )
        if self.options.with_sycl:
            implementation = str(self.options.sycl_implementation)
            toolchain.cache_variables["SYCLFFT_SYCL_IMPLEMENTATION"] = (
                "AdaptiveCpp" if implementation == "adaptivecpp" else "DPCPP"
            )
            cufft = self.options.get_safe("with_cufft", "auto")
            cufft_text = str(cufft).lower()
            if cufft_text == "true":
                cufft_value = "ON"
            elif cufft_text == "false":
                cufft_value = "OFF"
            else:
                cufft_value = "AUTO"
            toolchain.cache_variables["SYCLFFT_CUFFT"] = cufft_value
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if not self.conf.get("tools.build:skip_test", default=True, check_type=bool):
            cmake.ctest()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        copy(self, "NOTICE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        copy(self, "LLVM-EXCEPTION.txt",
             src=os.path.join(self.source_folder, "third_party", "syclcplx"),
             dst=os.path.join(self.package_folder, "licenses"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "syclfft")
        self.cpp_info.components["core"].set_property("cmake_target_name", "syclfft::core")
        self.cpp_info.components["core"].libs = ["syclfft_core"]
        if self.options.with_host:
            host = self.cpp_info.components["host"]
            host.set_property("cmake_target_name", "syclfft::host")
            host.libs = ["syclfft_host"]
            host.requires = ["core"]
        if self.options.with_sycl:
            sycl = self.cpp_info.components["sycl"]
            sycl.set_property("cmake_target_name", "syclfft::sycl")
            sycl.libs = ["syclfft_sycl"]
            sycl.requires = ["core"]
            if self.options.with_host:
                sycl.requires.append("host")
        provider_root = "bin" if self.settings.os == "Windows" else "lib"
        provider_dir = os.path.join(
            self.package_folder, provider_root, "syclfft", "providers"
        )
        self.runenv_info.prepend_path("SYCLFFT_PLUGIN_PATH", provider_dir)
