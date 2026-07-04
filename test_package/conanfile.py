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
        self.run(os.path.join(self.cpp.build.bindir, "test_sycl"), env="conanrun")
