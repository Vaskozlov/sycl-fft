#include "vkfft_compiler.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <new>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>

namespace syclfft::detail
{
    namespace
    {
        void check_system(int error, const char *operation)
        {
            if (error != 0) {
                throw std::system_error(error, std::generic_category(), operation);
            }
        }

        class memory_file
        {
        public:
            explicit memory_file(const char *name)
            {
                fd_ = memfd_create(name, MFD_CLOEXEC);
                if (fd_ < 0) {
                    check_system(errno, "memfd_create");
                }
                if (fd_ <= STDERR_FILENO) {
                    const int duplicate = fcntl(fd_, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
                    const int error = errno;
                    close(fd_);
                    fd_ = duplicate;
                    if (fd_ < 0) {
                        check_system(error, "fcntl");
                    }
                }
            }

            ~memory_file()
            {
                close(fd_);
            }

            memory_file(const memory_file &) = delete;
            memory_file &operator=(const memory_file &) = delete;

            int fd() const
            {
                return fd_;
            }

            void rewind() const
            {
                if (lseek(fd_, 0, SEEK_SET) < 0) {
                    check_system(errno, "lseek");
                }
            }

            void write_source(const char *data) const
            {
                auto remaining = std::strlen(data);
                while (remaining > 0) {
                    const auto count = write(fd_, data, remaining);
                    if (count < 0 && errno == EINTR) {
                        continue;
                    }
                    if (count <= 0) {
                        check_system(count == 0 ? EIO : errno, "write");
                    }
                    data += count;
                    remaining -= static_cast<std::size_t>(count);
                }
                rewind();
            }

            std::string read_at_most(std::size_t limit) const
            {
                struct stat status{};
                if (fstat(fd_, &status) != 0) {
                    check_system(errno, "fstat");
                }
                rewind();
                std::string data(std::min(limit, static_cast<std::size_t>(status.st_size)), '\0');
                std::size_t offset = 0;
                while (offset < data.size()) {
                    const auto count = read(fd_, data.data() + offset, data.size() - offset);
                    if (count < 0 && errno == EINTR) {
                        continue;
                    }
                    if (count <= 0) {
                        check_system(count == 0 ? EIO : errno, "read");
                    }
                    offset += static_cast<std::size_t>(count);
                }
                return data;
            }

        private:
            int fd_{-1};
        };

        class spawn_actions
        {
        public:
            spawn_actions()
            {
                check_system(
                    posix_spawn_file_actions_init(&actions_), "posix_spawn_file_actions_init");
            }

            ~spawn_actions()
            {
                posix_spawn_file_actions_destroy(&actions_);
            }

            spawn_actions(const spawn_actions &) = delete;
            spawn_actions &operator=(const spawn_actions &) = delete;

            void redirect(int from, int to)
            {
                check_system(
                    posix_spawn_file_actions_adddup2(&actions_, from, to),
                    "posix_spawn_file_actions_adddup2");
            }

            void close_other_files()
            {
#if defined(__GLIBC_PREREQ)
#    if __GLIBC_PREREQ(2, 34)
                check_system(
                    posix_spawn_file_actions_addclosefrom_np(&actions_, STDERR_FILENO + 1),
                    "posix_spawn_file_actions_addclosefrom_np");
#    endif
#endif
            }

            const posix_spawn_file_actions_t *get() const
            {
                return &actions_;
            }

        private:
            posix_spawn_file_actions_t actions_{};
        };

        const std::string &compiler_path()
        {
            static const auto path = [] {
                Dl_info provider{};
                if (dladdr(reinterpret_cast<const void *>(&compile_cuda_ptx), &provider) == 0
                    || !provider.dli_fname) {
                    throw std::runtime_error("could not locate the VkFFT provider");
                }
                const auto directory = std::filesystem::canonical(provider.dli_fname).parent_path();
                return (directory / "syclfft-vkfft-compiler").string();
            }();
            return path;
        }

        int compile_in_helper(
            const char *source, unsigned major, unsigned minor, int disable_fma,
            const memory_file &assembly, const memory_file &diagnostics)
        {
            const memory_file input("syclfft-vkfft-source");
            input.write_source(source);
            spawn_actions actions;
            actions.redirect(input.fd(), STDIN_FILENO);
            actions.redirect(assembly.fd(), STDOUT_FILENO);
            actions.redirect(diagnostics.fd(), STDERR_FILENO);
            actions.close_other_files();

            auto path = compiler_path();
            auto architecture = "sm_" + std::to_string(major) + std::to_string(minor);
            char fma_flag[] = {disable_fma ? '1' : '0', '\0'};
            char *argv[] = {path.data(), architecture.data(), fma_flag, nullptr};
            // Retain library discovery for relocatable LLVM installations, but exclude
            // preloaded libraries which may register their own Clang options in the helper.
            const auto *library_path = std::getenv("LD_LIBRARY_PATH");
            std::string library_environment;
            if (library_path) {
                library_environment = std::string("LD_LIBRARY_PATH=") + library_path;
            }
            char *environment[] = {
                library_environment.empty() ? nullptr : library_environment.data(), nullptr};
            pid_t child{};
            check_system(
                posix_spawn(&child, path.c_str(), actions.get(), nullptr, argv, environment),
                "could not start syclfft-vkfft-compiler");

            int status{};
            while (waitpid(child, &status, 0) < 0) {
                if (errno == ECHILD) {
                    return -1;
                }
                if (errno != EINTR) {
                    check_system(errno, "waitpid");
                }
            }
            return status;
        }

        void report_error(std::string *diagnostics, const char *message) noexcept
        {
            if (diagnostics) {
                try {
                    *diagnostics = message;
                } catch (...) {
                }
            }
        }
    } // namespace

    void require_cuda_ptx_compiler()
    {
        const auto &path = compiler_path();
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("VkFFT compiler helper is missing: " + path);
        }
        if (access(path.c_str(), X_OK) != 0) {
            const auto error = errno;
            throw std::system_error(
                error, std::generic_category(), "VkFFT compiler helper is not executable: " + path);
        }
    }

    int compile_cuda_ptx(
        void *diagnostics, const char *source, unsigned major, unsigned minor, int disable_fma,
        char **ptx, std::size_t *ptx_size) noexcept
    {
        auto *error = static_cast<std::string *>(diagnostics);
        if (!ptx || !ptx_size || !source) {
            report_error(error, "invalid VkFFT compilation argument");
            return 1;
        }
        *ptx = nullptr;
        *ptx_size = 0;
        try {
            const memory_file assembly("syclfft-vkfft-ptx");
            const memory_file log("syclfft-vkfft-diagnostics");
            const int status = compile_in_helper(source, major, minor, disable_fma, assembly, log);
            const bool exited_successfully =
                status < 0 || (WIFEXITED(status) && WEXITSTATUS(status) == 0);
            const auto text = exited_successfully ? assembly.read_at_most(std::string{}.max_size())
                                                  : std::string{};
            // A host SIGCHLD handler can reap the helper first. Its completion marker
            // distinguishes fully generated PTX from a partial write in that case.
            if (text.size() < 2 || text.back() != '\0') {
                auto message = log.read_at_most(4096);
                if (status >= 0 && WIFSIGNALED(status)) {
                    message +=
                        "\nVkFFT compiler terminated by signal " + std::to_string(WTERMSIG(status));
                } else if (message.empty()) {
                    message = status >= 0 && WIFEXITED(status) && WEXITSTATUS(status) != 0
                                  ? "VkFFT compiler returned status "
                                        + std::to_string(WEXITSTATUS(status))
                                  : "VkFFT compiler returned incomplete PTX";
                }
                report_error(error, message.c_str());
                return 1;
            }
            auto *output = static_cast<char *>(std::malloc(text.size()));
            if (!output) {
                throw std::bad_alloc{};
            }
            std::memcpy(output, text.data(), text.size());
            *ptx = output;
            *ptx_size = text.size();
            return 0;
        } catch (const std::exception &failure) {
            report_error(error, failure.what());
        } catch (...) {
            report_error(error, "unexpected VkFFT compiler failure");
        }
        return 1;
    }
} // namespace syclfft::detail
