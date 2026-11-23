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
        if dependency.options.build_sycl:
            implementation = dependency.options.sycl_implementation
            toolchain.cache_variables["SYCLFFT_TEST_IMPLEMENTATION"] = (
                "AdaptiveCpp" if implementation == "adaptivecpp" else "DPCPP"
            )
        paths = {
            "adaptivecpp_root": "AdaptiveCpp_ROOT",
            "adaptivecpp_version": "SYCLFFT_EXPECTED_ACPP_VERSION",
            "acpp_targets": "ACPP_TARGETS",
        }
        for option, variable in paths.items():
            value = dependency.options.get_safe(option)
            if value:
                toolchain.cache_variables[variable] = str(value).replace("\\", "/")
        toolchain.generate()
        CMakeDeps(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if not can_run(self):
            return
        target = "test_sycl" if self.dependencies["syclfft"].options.build_sycl else "test_core"
        self.run(os.path.join(self.cpp.build.bindir, target), env="conanrun")
