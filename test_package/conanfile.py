from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.build import can_run
import os


class SyclFftTestPackage(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    test_type = "explicit"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        dependency = self.dependencies["syclfft"]
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["SYCLFFT_TEST_HOST"] = bool(
            dependency.options.with_host
        )
        toolchain.cache_variables["SYCLFFT_TEST_SYCL"] = bool(
            dependency.options.with_sycl
        )
        implementation = dependency.options.get_safe("sycl_implementation", "dpcpp")
        toolchain.cache_variables["SYCLFFT_TEST_IMPLEMENTATION"] = (
            "AdaptiveCpp" if str(implementation) == "adaptivecpp" else "DPCPP"
        )
        toolchain.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if not can_run(self):
            return
        dependency = self.dependencies["syclfft"]
        if dependency.options.with_host:
            self.run(os.path.join(self.cpp.build.bindir, "test_host"), env="conanrun")
        if dependency.options.with_sycl:
            self.run(os.path.join(self.cpp.build.bindir, "test_sycl"), env="conanrun")
