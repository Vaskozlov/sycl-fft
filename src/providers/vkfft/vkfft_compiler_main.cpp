#include <clang/Basic/Diagnostic.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <exception>
#include <iostream>
#include <iterator>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>
#include <string_view>
#include <sys/resource.h>

namespace
{
    // The CUDA spelling used by VkFFT is lowered by Clang's NVPTX frontend
    // without including CUDA's toolkit headers.
    constexpr char cuda_prelude[] = R"CUDA(
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared, aligned(16)))
#define __device__ __attribute__((device))
#define __constant__ __attribute__((constant))
#define __launch_bounds__(n) __attribute__((launch_bounds(n)))

struct alignas(8) float2 { float x, y; };
struct alignas(16) double2 { double x, y; };
struct alignas(8) uint2 { unsigned x, y; };
struct alignas(8) int2 { int x, y; };

__device__ inline float2 make_float2(float x, float y) { return {x, y}; }
__device__ inline double2 make_double2(double x, double y) { return {x, y}; }

struct thread_index {
    __declspec(property(get=get_x)) unsigned x;
    __declspec(property(get=get_y)) unsigned y;
    __declspec(property(get=get_z)) unsigned z;
    __device__ static unsigned get_x() { return __nvvm_read_ptx_sreg_tid_x(); }
    __device__ static unsigned get_y() { return __nvvm_read_ptx_sreg_tid_y(); }
    __device__ static unsigned get_z() { return __nvvm_read_ptx_sreg_tid_z(); }
};

struct block_index {
    __declspec(property(get=get_x)) unsigned x;
    __declspec(property(get=get_y)) unsigned y;
    __declspec(property(get=get_z)) unsigned z;
    __device__ static unsigned get_x() { return __nvvm_read_ptx_sreg_ctaid_x(); }
    __device__ static unsigned get_y() { return __nvvm_read_ptx_sreg_ctaid_y(); }
    __device__ static unsigned get_z() { return __nvvm_read_ptx_sreg_ctaid_z(); }
};

extern const __device__ thread_index threadIdx;
extern const __device__ block_index blockIdx;

__device__ inline float fmaf(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
__device__ inline float fma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
__device__ inline double fma(double a, double b, double c) { return __builtin_fma(a, b, c); }

__device__ inline void __sincosf(float a, float *s, float *c)
{
    asm("sin.approx.f32 %0,%1;" : "=f"(*s) : "f"(a));
    asm("cos.approx.f32 %0,%1;" : "=f"(*c) : "f"(a));
}
)CUDA";

    void record_diagnostic(std::string *diagnostics, std::string_view message) noexcept
    {
        try {
            if (diagnostics && diagnostics->size() < 4096) {
                if (!diagnostics->empty()) {
                    diagnostics->push_back('\n');
                }
                diagnostics->append(message.substr(0, 4096 - diagnostics->size()));
            }
        } catch (...) {
        }
    }

    class compiler_diagnostics : public clang::DiagnosticConsumer
    {
    public:
        explicit compiler_diagnostics(std::string *diagnostics)
          : diagnostics_(diagnostics)
        {}

        void HandleDiagnostic(
            clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) override
        {
            clang::DiagnosticConsumer::HandleDiagnostic(level, info);
            if (level >= clang::DiagnosticsEngine::Error) {
                llvm::SmallString<256> message;
                info.FormatDiagnostic(message);
                record_diagnostic(diagnostics_, {message.data(), message.size()});
            }
        }

    private:
        std::string *diagnostics_;
    };

    bool emit_ptx(
        const char *source, const std::string &architecture, bool disable_fma,
        std::string *diagnostics, llvm::SmallVectorImpl<char> &ptx)
    {
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();

        constexpr char triple[] = "nvptx64-nvidia-cuda";
        std::string target_error;
        const auto *target = llvm::TargetRegistry::lookupTarget(triple, target_error);
        if (!target) {
            record_diagnostic(diagnostics, target_error);
            return false;
        }
        const std::unique_ptr<llvm::MCSubtargetInfo> subtarget{
            target->createMCSubtargetInfo(triple, "", "")};
        if (!subtarget || !subtarget->isCPUStringValid(architecture)) {
            record_diagnostic(
                diagnostics, "LLVM does not support CUDA architecture " + architecture);
            return false;
        }

        constexpr char source_path[] = "/VkFFT-generated.cu";
        auto filesystem = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
        filesystem->addFile(
            source_path,
            0,
            llvm::MemoryBuffer::getMemBufferCopy(
                std::string(cuda_prelude) + "\n#line 1 \"VkFFT-generated.cu\"\n" + source));

        auto invocation = std::make_shared<clang::CompilerInvocation>();
        compiler_diagnostics diagnostic_consumer(diagnostics);
        clang::CompilerInstance compiler;
        compiler.setInvocation(invocation);
        compiler.createDiagnostics(*filesystem, &diagnostic_consumer, false);

        const char *arguments[] = {
            "-triple",
            triple,
            "-target-cpu",
            architecture.c_str(),
            "-fcuda-is-device",
            "-fdeclspec",
            "-nostdsysteminc",
            "-nobuiltininc",
            disable_fma ? "-ffp-contract=off" : "-ffp-contract=fast",
            "-O2",
            "-S",
            "-x",
            "cuda",
            source_path,
        };
        if (!clang::CompilerInvocation::CreateFromArgs(
                *invocation, arguments, compiler.getDiagnostics())) {
            return false;
        }

        compiler.createFileManager(filesystem);
        compiler.setOutputStream(std::make_unique<llvm::raw_svector_ostream>(ptx));
        clang::EmitAssemblyAction action;
        return compiler.ExecuteAction(action) && !ptx.empty();
    }
} // namespace

int main(int argc, char **argv)
try {
    const rlimit no_core_dump{0, 0};
    (void)setrlimit(RLIMIT_CORE, &no_core_dump);
    if (argc != 3) {
        std::cerr << "invalid VkFFT compiler invocation\n";
        return 1;
    }

    std::ios::sync_with_stdio(false);
    const std::string source{std::istreambuf_iterator<char>(std::cin), {}};
    llvm::SmallVector<char, 4096> ptx;
    std::string diagnostics;
    const std::string architecture = argv[1];
    if (!emit_ptx(
            source.c_str(), architecture, std::string_view(argv[2]) == "1", &diagnostics, ptx)) {
        std::cerr << "Clang could not generate PTX for " << architecture << '\n'
                  << diagnostics << '\n';
        return 1;
    }
    std::cout.write(ptx.data(), static_cast<std::streamsize>(ptx.size()));
    std::cout.put('\0');
    std::cout.flush();
    return std::cout ? 0 : 1;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
