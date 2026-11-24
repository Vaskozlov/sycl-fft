import os

from conan import ConanFile
from conan.tools.build import can_run, cmd_args_to_string
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


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
        targets = dependency.options.get_safe("acpp_targets")
        if targets:
            toolchain.cache_variables["ACPP_TARGETS"] = str(targets)
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
        self.run(cmd_args_to_string([os.path.join(self.cpp.build.bindirs[0], target)]),
                 env="conanrun")
