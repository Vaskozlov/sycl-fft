import os

from conan import ConanFile
from conan.tools.files import apply_conandata_patches, copy, export_conandata_patches, get


class VkFFTConan(ConanFile):
    name = "vkfft-driver"
    version = "1.3.4"
    description = "VkFFT headers patched for toolkit-free syclfft CUDA plans"
    url = "https://github.com/DTolm/VkFFT"
    license = "MIT"
    package_type = "header-library"
    no_copy_source = True

    def export_sources(self):
        export_conandata_patches(self)

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)
        apply_conandata_patches(self)

    def package(self):
        headers = os.path.join(self.source_folder, "vkFFT")
        copy(self, "*.h", src=headers, dst=os.path.join(self.package_folder, "include"))
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))

    def package_info(self):
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
