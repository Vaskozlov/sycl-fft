import os
import re

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd, cross_building
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


class SyclFftConan(ConanFile):
    name = "syclfft"
    version = "0.1.0"
    package_type = "shared-library"
    license = "Apache-2.0"
    url = "https://github.com/vaskozlov/sycl-fft"
    description = "FFTW-like reusable C++ FFT plans for SYCL USM"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "build_sycl": [True, False],
        "build_tests": [True, False],
        "build_examples": [True, False],
        "build_benchmarks": [True, False],
        "sycl_implementation": ["adaptivecpp", "dpcpp"],
        "adaptivecpp_root": [None, "ANY"],
        "adaptivecpp_version": [None, "ANY"],
        "acpp_targets": [None, "ANY"],
        "with_fftw": [True, False],
        "use_system_fftw": [True, False],
        "fftw_version": ["ANY"],
        "fftw_root": [None, "ANY"],
        "with_cufft": ["auto", True, False],
        "cuda_toolkit_version": [None, "ANY"],
        "cuda_toolkit_root": [None, "ANY"],
        "with_vkfft": [True, False],
        "llvm_version": ["ANY"],
        "llvm_dir": [None, "ANY"],
        "clang_dir": [None, "ANY"],
        "cuda_driver_include_dir": [None, "ANY"],
    }
    default_options = {
        "build_sycl": True,
        "build_tests": False,
        "build_examples": True,
        "build_benchmarks": False,
        "sycl_implementation": "adaptivecpp",
        "adaptivecpp_root": None,
        "adaptivecpp_version": None,
        "acpp_targets": None,
        "with_fftw": True,
        "use_system_fftw": False,
        "fftw_version": "3.3.10",
        "fftw_root": None,
        "with_cufft": "auto",
        "cuda_toolkit_version": None,
        "cuda_toolkit_root": None,
        "with_vkfft": False,
        "llvm_version": "20.1.8",
        "llvm_dir": None,
        "clang_dir": None,
        "cuda_driver_include_dir": None,
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
        "examples/*", "tests/*", "benchmarks/*", "LICENSE", "NOTICE", "README.md",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.sycl_implementation = "dpcpp"

    def configure(self):
        if not self.options.with_fftw:
            self.options.rm_safe("use_system_fftw")
            self.options.rm_safe("fftw_version")
        if not self.options.get_safe("use_system_fftw"):
            self.options.rm_safe("fftw_root")
        if not self._with_cufft:
            self.options.rm_safe("cuda_toolkit_root")
            self.options.rm_safe("cuda_toolkit_version")
        if not self.options.with_vkfft:
            self.options.rm_safe("llvm_version")
            self.options.rm_safe("llvm_dir")
            self.options.rm_safe("clang_dir")
            self.options.rm_safe("cuda_driver_include_dir")
        if not self._with_adaptivecpp:
            self.options.rm_safe("adaptivecpp_root")
            self.options.rm_safe("adaptivecpp_version")
            self.options.rm_safe("acpp_targets")

    @property
    def _with_adaptivecpp(self):
        return self.options.build_sycl and self.options.sycl_implementation == "adaptivecpp"

    @property
    def _with_cufft(self):
        if not self.options.build_sycl:
            return False
        if self.options.with_cufft == "auto":
            return not self.options.with_vkfft and bool(self.options.get_safe("cuda_toolkit_version"))
        return bool(self.options.with_cufft)

    def requirements(self):
        if self.options.with_fftw:
            version = self.options.fftw_version
            if self.options.use_system_fftw:
                self.requires(f"fftw-system/{version}", headers=False, libs=False, run=True,
                              package_id_mode="full_mode")
            else:
                self.requires(f"fftw/{version}")
        if self._with_cufft and self.options.get_safe("cuda_toolkit_version"):
            self.requires(f"cuda-toolkit-system/{self.options.cuda_toolkit_version}",
                          headers=False, libs=False, run=True, package_id_mode="full_mode")
        if self.options.with_vkfft:
            self.requires("vkfft-driver/1.3.4", transitive_headers=False, package_id_mode="full_mode")
            self.requires(f"clang-cpp-system/{self.options.llvm_version}",
                          headers=False, libs=False, run=True, package_id_mode="full_mode")

    def build_requirements(self):
        # Development packages belong only to machines building syclfft from source.
        if cross_building(self):
            # Cross profiles supply target development files through their sysroot.
            return
        if self.options.with_vkfft:
            self.tool_requires("clang-cpp-system/<host_version>",
                               options={"development": True})
        if self.options.get_safe("use_system_fftw"):
            self.tool_requires("fftw-system/<host_version>",
                               options={"development": True})
        if self._with_cufft and self.options.get_safe("cuda_toolkit_version"):
            self.tool_requires("cuda-toolkit-system/<host_version>",
                               options={"development": True})

    def validate(self):
        check_min_cppstd(self, "17")
        if self.options.build_benchmarks and not self.options.build_sycl:
            raise ConanInvalidConfiguration("build_benchmarks requires build_sycl")
        if self.options.with_cufft != "auto" and self.options.with_cufft:
            if not self.options.build_sycl:
                raise ConanInvalidConfiguration("with_cufft requires build_sycl")
            if not re.fullmatch(r"\d+\.\d+\.\d+", str(self.options.get_safe("cuda_toolkit_version"))):
                raise ConanInvalidConfiguration("with_cufft requires an exact cuda_toolkit_version")
        if self.options.with_vkfft:
            if self.settings.os != "Linux" or not self._with_adaptivecpp:
                raise ConanInvalidConfiguration("with_vkfft requires Linux and AdaptiveCpp SYCL")
            if not re.fullmatch(r"20\.\d+\.\d+", str(self.options.llvm_version)):
                raise ConanInvalidConfiguration("with_vkfft requires an exact LLVM 20 llvm_version")
            if cross_building(self):
                raise ConanInvalidConfiguration("with_vkfft requires a native build")

    def package_id(self):
        # These paths select installations of versions already declared in the graph.
        for path in ("fftw_root", "cuda_toolkit_root", "llvm_dir", "clang_dir"):
            self.info.options.rm_safe(path)

    def layout(self):
        cmake_layout(self)

    def _set_package_locations(self, cache):
        paths = {
            "adaptivecpp_root": "AdaptiveCpp_ROOT",
            "fftw_root": "FFTW3_ROOT",
            "cuda_toolkit_root": "CUDAToolkit_ROOT",
            "llvm_dir": "LLVM_DIR",
            "clang_dir": "Clang_DIR",
            "cuda_driver_include_dir": "SYCLFFT_CUDA_DRIVER_INCLUDE_DIR",
        }
        for option, variable in paths.items():
            value = self.options.get_safe(option)
            if value:
                cache[variable] = str(value).replace("\\", "/")

    def generate(self):
        CMakeDeps(self).generate()
        toolchain = CMakeToolchain(self)
        cache = toolchain.cache_variables
        cache["BUILD_SHARED_LIBS"] = True
        cache["SYCLFFT_BUILD_SYCL"] = bool(self.options.build_sycl)
        cache["SYCLFFT_BUILD_TESTS"] = bool(self.options.build_tests)
        cache["SYCLFFT_BUILD_EXAMPLES"] = bool(self.options.build_examples)
        cache["SYCLFFT_BUILD_BENCHMARKS"] = bool(self.options.build_benchmarks)
        cache["SYCLFFT_FFTW"] = "ON" if self.options.with_fftw else "OFF"
        cache["SYCLFFT_USE_SYSTEM_FFTW"] = bool(self.options.get_safe("use_system_fftw"))
        cache["SYCLFFT_SYCL_IMPLEMENTATION"] = (
            "AdaptiveCpp" if self._with_adaptivecpp else "DPCPP"
        )
        cache["SYCLFFT_CUFFT"] = "ON" if self._with_cufft else "OFF"
        cache["SYCLFFT_VKFFT"] = bool(self.options.with_vkfft)

        self._set_package_locations(cache)
        if self._with_adaptivecpp:
            if self.options.adaptivecpp_version:
                cache["SYCLFFT_EXPECTED_ACPP_VERSION"] = str(self.options.adaptivecpp_version)
            if self.options.acpp_targets:
                cache["ACPP_TARGETS"] = str(self.options.acpp_targets)
        if self.options.get_safe("use_system_fftw"):
            cache["SYCLFFT_EXPECTED_FFTW_VERSION"] = str(self.dependencies["fftw-system"].ref.version)
        if self._with_cufft:
            cache["SYCLFFT_EXPECTED_CUDATOOLKIT_VERSION"] = str(
                self.dependencies["cuda-toolkit-system"].ref.version
            )
        if self.options.with_vkfft:
            cache["SYCLFFT_EXPECTED_LLVM_VERSION"] = str(self.dependencies["clang-cpp-system"].ref.version)
            cache["SYCLFFT_VKFFT_INCLUDE_DIR"] = self.dependencies["vkfft-driver"].cpp_info.includedirs[0]
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if self.options.build_tests and not self.conf.get(
            "tools.build:skip_test", default=False, check_type=bool
        ):
            cmake.test()

    def package(self):
        CMake(self).install()
        license_dir = os.path.join(self.package_folder, "licenses")
        for filename in ("LICENSE", "NOTICE"):
            copy(self, filename, src=self.source_folder, dst=license_dir)
        copy(self, "*.txt", src=os.path.join(self.package_folder, "share", "syclfft", "licenses"),
             dst=license_dir)

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "syclfft")
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["lib/cmake/syclfft"]
        core = self.cpp_info.components["core"]
        core.set_property("cmake_target_name", "syclfft::core")
        core.builddirs = self.cpp_info.builddirs
        core.libs = ["syclfft_core"]
        if self.options.build_sycl:
            sycl = self.cpp_info.components["sycl"]
            sycl.set_property("cmake_target_name", "syclfft::sycl")
            sycl.builddirs = self.cpp_info.builddirs
            sycl.libs = ["syclfft_sycl"]
            sycl.requires = ["core"]
        provider_root = "bin" if self.settings.os == "Windows" else "lib"
        provider_dir = os.path.join(self.package_folder, provider_root, "syclfft", "providers")
        self.runenv_info.prepend_path("SYCLFFT_PLUGIN_PATH", provider_dir)
