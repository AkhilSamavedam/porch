#include "porch/cuda_jit.hpp"

#include <dlfcn.h>

#include <initializer_list>
#include <memory>
#include <mutex>
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <unordered_map>
#include <utility>

namespace porch::cuda_jit {
    namespace {

        using nvrtc_program = void*;
        using nvrtc_result = int;
        using cuda_device = int;
        using cuda_context = void*;
        using cuda_device_ptr = uint64_t;
        using cuda_module = void*;
        using cuda_function = void*;
        using cuda_stream = void*;
        using cuda_result = int;

        constexpr nvrtc_result nvrtc_success = 0;
        constexpr cuda_result cuda_success = 0;

        class dynamic_library {
          public:
            dynamic_library() = default;

            explicit dynamic_library(const char* name) {
                handle_ = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
            }

            dynamic_library(const dynamic_library&) = delete;
            dynamic_library& operator=(const dynamic_library&) = delete;

            dynamic_library(dynamic_library&& other) noexcept
                : handle_(std::exchange(other.handle_, nullptr)) {}

            dynamic_library& operator=(dynamic_library&& other) noexcept {
                if (this == &other) return *this;
                reset();
                handle_ = std::exchange(other.handle_, nullptr);
                return *this;
            }

            ~dynamic_library() { reset(); }

            [[nodiscard]] explicit operator bool() const noexcept {
                return handle_ != nullptr;
            }

            template <typename Function>
            [[nodiscard]] Function symbol(const char* name) const noexcept {
                return reinterpret_cast<Function>(dlsym(handle_, name));
            }

          private:
            void reset() noexcept {
                if (handle_ == nullptr) return;
                dlclose(handle_);
                handle_ = nullptr;
            }

            void* handle_ = nullptr;
        };

        [[nodiscard]] dynamic_library open_any(
            std::initializer_list<const char*> names) {
            for (const char* name : names) {
                dynamic_library library{name};
                if (library) return library;
            }
            return {};
        }

        [[nodiscard]] bool cuda_driver_available() noexcept {
            return static_cast<bool>(open_any({"libcuda.so.1", "libcuda.so"}));
        }

        struct nvrtc_api {
            using create_program_fn = nvrtc_result (*)(nvrtc_program*,
                                                       const char*, const char*,
                                                       int, const char* const*,
                                                       const char* const*);
            using compile_program_fn = nvrtc_result (*)(nvrtc_program, int,
                                                        const char* const*);
            using get_ptx_size_fn = nvrtc_result (*)(nvrtc_program, size_t*);
            using get_ptx_fn = nvrtc_result (*)(nvrtc_program, char*);
            using get_log_size_fn = nvrtc_result (*)(nvrtc_program, size_t*);
            using get_log_fn = nvrtc_result (*)(nvrtc_program, char*);
            using destroy_program_fn = nvrtc_result (*)(nvrtc_program*);
            using get_error_string_fn = const char* (*)(nvrtc_result);

            create_program_fn create_program = nullptr;
            compile_program_fn compile_program = nullptr;
            get_ptx_size_fn get_ptx_size = nullptr;
            get_ptx_fn get_ptx = nullptr;
            get_log_size_fn get_log_size = nullptr;
            get_log_fn get_log = nullptr;
            destroy_program_fn destroy_program = nullptr;
            get_error_string_fn get_error_string = nullptr;
        };

        struct nvrtc_runtime {
            dynamic_library library;
            nvrtc_api api;

            [[nodiscard]] explicit operator bool() const noexcept {
                return static_cast<bool>(library) &&
                       api.create_program != nullptr &&
                       api.compile_program != nullptr &&
                       api.get_ptx_size != nullptr && api.get_ptx != nullptr &&
                       api.get_log_size != nullptr && api.get_log != nullptr &&
                       api.destroy_program != nullptr &&
                       api.get_error_string != nullptr;
            }
        };

        struct cuda_api {
            using init_fn = cuda_result (*)(uint32_t);
            using device_get_fn = cuda_result (*)(cuda_device*, int);
            using ctx_create_fn = cuda_result (*)(cuda_context*, uint32_t,
                                                  cuda_device);
            using ctx_set_current_fn = cuda_result (*)(cuda_context);
            using ctx_destroy_fn = cuda_result (*)(cuda_context);
            using stream_create_fn = cuda_result (*)(cuda_stream*, uint32_t);
            using stream_destroy_fn = cuda_result (*)(cuda_stream);
            using stream_synchronize_fn = cuda_result (*)(cuda_stream);
            using mem_alloc_fn = cuda_result (*)(cuda_device_ptr*, size_t);
            using mem_free_fn = cuda_result (*)(cuda_device_ptr);
            using memcpy_htod_fn = cuda_result (*)(cuda_device_ptr, const void*,
                                                   size_t);
            using memcpy_dtoh_fn = cuda_result (*)(void*, cuda_device_ptr,
                                                   size_t);
            using module_load_data_fn = cuda_result (*)(cuda_module*,
                                                        const void*);
            using module_unload_fn = cuda_result (*)(cuda_module);
            using module_get_function_fn = cuda_result (*)(cuda_function*,
                                                           cuda_module,
                                                           const char*);
            using launch_kernel_fn = cuda_result (*)(
                cuda_function, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                uint32_t, uint32_t, cuda_stream, void**, void**);
            using ctx_synchronize_fn = cuda_result (*)();
            using get_error_name_fn = cuda_result (*)(cuda_result,
                                                      const char**);
            using get_error_string_fn = cuda_result (*)(cuda_result,
                                                        const char**);

            init_fn init = nullptr;
            device_get_fn device_get = nullptr;
            ctx_create_fn ctx_create = nullptr;
            ctx_set_current_fn ctx_set_current = nullptr;
            ctx_destroy_fn ctx_destroy = nullptr;
            stream_create_fn stream_create = nullptr;
            stream_destroy_fn stream_destroy = nullptr;
            stream_synchronize_fn stream_synchronize = nullptr;
            mem_alloc_fn mem_alloc = nullptr;
            mem_free_fn mem_free = nullptr;
            memcpy_htod_fn memcpy_htod = nullptr;
            memcpy_dtoh_fn memcpy_dtoh = nullptr;
            module_load_data_fn module_load_data = nullptr;
            module_unload_fn module_unload = nullptr;
            module_get_function_fn module_get_function = nullptr;
            launch_kernel_fn launch_kernel = nullptr;
            ctx_synchronize_fn ctx_synchronize = nullptr;
            get_error_name_fn get_error_name = nullptr;
            get_error_string_fn get_error_string = nullptr;
        };

        struct cuda_runtime {
            dynamic_library library;
            cuda_api api;

            [[nodiscard]] explicit operator bool() const noexcept {
                return static_cast<bool>(library) && api.init != nullptr &&
                       api.device_get != nullptr && api.ctx_create != nullptr &&
                       api.ctx_set_current != nullptr &&
                       api.ctx_destroy != nullptr && api.mem_alloc != nullptr &&
                       api.stream_create != nullptr &&
                       api.stream_destroy != nullptr &&
                       api.stream_synchronize != nullptr &&
                       api.mem_free != nullptr && api.memcpy_htod != nullptr &&
                       api.memcpy_dtoh != nullptr &&
                       api.module_load_data != nullptr &&
                       api.module_unload != nullptr &&
                       api.module_get_function != nullptr &&
                       api.launch_kernel != nullptr &&
                       api.ctx_synchronize != nullptr &&
                       api.get_error_name != nullptr &&
                       api.get_error_string != nullptr;
            }
        };

        [[nodiscard]] nvrtc_runtime load_nvrtc() {
            dynamic_library library =
                open_any({"libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11.2"});
            if (!library) return {};

            nvrtc_api api;
            api.create_program = library.symbol<nvrtc_api::create_program_fn>(
                "nvrtcCreateProgram");
            api.compile_program = library.symbol<nvrtc_api::compile_program_fn>(
                "nvrtcCompileProgram");
            api.get_ptx_size =
                library.symbol<nvrtc_api::get_ptx_size_fn>("nvrtcGetPTXSize");
            api.get_ptx = library.symbol<nvrtc_api::get_ptx_fn>("nvrtcGetPTX");
            api.get_log_size = library.symbol<nvrtc_api::get_log_size_fn>(
                "nvrtcGetProgramLogSize");
            api.get_log =
                library.symbol<nvrtc_api::get_log_fn>("nvrtcGetProgramLog");
            api.destroy_program = library.symbol<nvrtc_api::destroy_program_fn>(
                "nvrtcDestroyProgram");
            api.get_error_string =
                library.symbol<nvrtc_api::get_error_string_fn>(
                    "nvrtcGetErrorString");

            return {std::move(library), api};
        }

        [[nodiscard]] cuda_runtime load_cuda_driver() {
            dynamic_library library = open_any({"libcuda.so.1", "libcuda.so"});
            if (!library) return {};

            cuda_api api;
            api.init = library.symbol<cuda_api::init_fn>("cuInit");
            api.device_get =
                library.symbol<cuda_api::device_get_fn>("cuDeviceGet");
            api.ctx_create =
                library.symbol<cuda_api::ctx_create_fn>("cuCtxCreate_v2");
            api.ctx_set_current =
                library.symbol<cuda_api::ctx_set_current_fn>("cuCtxSetCurrent");
            api.ctx_destroy =
                library.symbol<cuda_api::ctx_destroy_fn>("cuCtxDestroy_v2");
            api.stream_create =
                library.symbol<cuda_api::stream_create_fn>("cuStreamCreate");
            api.stream_destroy = library.symbol<cuda_api::stream_destroy_fn>(
                "cuStreamDestroy_v2");
            api.stream_synchronize =
                library.symbol<cuda_api::stream_synchronize_fn>(
                    "cuStreamSynchronize");
            api.mem_alloc =
                library.symbol<cuda_api::mem_alloc_fn>("cuMemAlloc_v2");
            api.mem_free =
                library.symbol<cuda_api::mem_free_fn>("cuMemFree_v2");
            api.memcpy_htod =
                library.symbol<cuda_api::memcpy_htod_fn>("cuMemcpyHtoD_v2");
            api.memcpy_dtoh =
                library.symbol<cuda_api::memcpy_dtoh_fn>("cuMemcpyDtoH_v2");
            api.module_load_data =
                library.symbol<cuda_api::module_load_data_fn>(
                    "cuModuleLoadData");
            api.module_unload =
                library.symbol<cuda_api::module_unload_fn>("cuModuleUnload");
            api.module_get_function =
                library.symbol<cuda_api::module_get_function_fn>(
                    "cuModuleGetFunction");
            api.launch_kernel =
                library.symbol<cuda_api::launch_kernel_fn>("cuLaunchKernel");
            api.ctx_synchronize = library.symbol<cuda_api::ctx_synchronize_fn>(
                "cuCtxSynchronize");
            api.get_error_name =
                library.symbol<cuda_api::get_error_name_fn>("cuGetErrorName");
            api.get_error_string =
                library.symbol<cuda_api::get_error_string_fn>(
                    "cuGetErrorString");

            return {std::move(library), api};
        }

        [[nodiscard]] std::string describe(const nvrtc_api& api,
                                           nvrtc_result result) {
            if (api.get_error_string == nullptr) return "unknown NVRTC error";
            return api.get_error_string(result);
        }

        void check_nvrtc(const nvrtc_api& api, nvrtc_result result,
                         std::string_view operation) {
            if (result == nvrtc_success) return;

            std::ostringstream message;
            message << operation << " failed: " << describe(api, result);
            throw std::runtime_error(message.str());
        }

        [[nodiscard]] std::string describe(const cuda_api& api,
                                           cuda_result result) {
            const char* name = nullptr;
            const char* description = nullptr;
            (void)api.get_error_name(result, &name);
            (void)api.get_error_string(result, &description);

            std::ostringstream message;
            if (name != nullptr) message << name;
            else message << "CUDA_ERROR_" << result;

            if (description != nullptr) message << ": " << description;
            return message.str();
        }

        void check_cuda(const cuda_api& api, cuda_result result,
                        std::string_view operation) {
            if (result == cuda_success) return;

            std::ostringstream message;
            message << operation << " failed: " << describe(api, result);
            throw std::runtime_error(message.str());
        }

        [[nodiscard]] std::string program_log(const nvrtc_api& api,
                                              nvrtc_program program) {
            size_t size = 0;
            check_nvrtc(api, api.get_log_size(program, &size),
                        "nvrtcGetProgramLogSize");
            if (size == 0) return {};

            std::string log(size, '\0');
            check_nvrtc(api, api.get_log(program, log.data()),
                        "nvrtcGetProgramLog");
            if (!log.empty() && log.back() == '\0') log.pop_back();
            return log;
        }

        [[nodiscard]] std::string cache_key(
            std::string_view source, const std::vector<std::string>& options) {
            std::ostringstream key;
            key << source.size() << ':' << source;
            for (const std::string& option : options) {
                key << '\0' << option.size() << ':' << option;
            }
            return key.str();
        }

        [[nodiscard]] std::unordered_map<std::string, std::string>& ptx_cache() {
            static std::unordered_map<std::string, std::string>* cache =
                new std::unordered_map<std::string, std::string>();
            return *cache;
        }

        [[nodiscard]] std::mutex& ptx_cache_mutex() {
            static std::mutex* mutex = new std::mutex();
            return *mutex;
        }

        class program_handle {
          public:
            program_handle(nvrtc_program program, nvrtc_api api) noexcept
                : program_(program), api_(api) {}

            program_handle(const program_handle&) = delete;
            program_handle& operator=(const program_handle&) = delete;

            ~program_handle() {
                if (program_ != nullptr) (void)api_.destroy_program(&program_);
            }

            [[nodiscard]] nvrtc_program get() const noexcept {
                return program_;
            }

          private:
            nvrtc_program program_;
            nvrtc_api api_;
        };

        class context_handle {
          public:
            context_handle(cuda_context context, cuda_api api) noexcept
                : context_(context), api_(api) {}

            context_handle(const context_handle&) = delete;
            context_handle& operator=(const context_handle&) = delete;

            ~context_handle() {
                if (context_ != nullptr) (void)api_.ctx_destroy(context_);
            }

          private:
            cuda_context context_;
            cuda_api api_;
        };

        class device_memory {
          public:
            device_memory(const cuda_api& api, size_t bytes)
                : api_(api), bytes_(bytes) {
                if (bytes_ != 0)
                    check_cuda(api_, api_.mem_alloc(&ptr_, bytes_),
                               "cuMemAlloc");
            }

            device_memory(const device_memory&) = delete;
            device_memory& operator=(const device_memory&) = delete;

            ~device_memory() {
                if (ptr_ != 0) (void)api_.mem_free(ptr_);
            }

            [[nodiscard]] cuda_device_ptr get() const noexcept { return ptr_; }

          private:
            cuda_api api_;
            cuda_device_ptr ptr_ = 0;
            size_t bytes_ = 0;
        };

        class module_handle {
          public:
            module_handle(cuda_module module, cuda_api api) noexcept
                : module_(module), api_(api) {}

            module_handle(const module_handle&) = delete;
            module_handle& operator=(const module_handle&) = delete;

            ~module_handle() {
                if (module_ != nullptr) (void)api_.module_unload(module_);
            }

            [[nodiscard]] cuda_module get() const noexcept { return module_; }

          private:
            cuda_module module_;
            cuda_api api_;
        };

        struct cuda_environment {
            cuda_environment() : runtime(load_cuda_driver()) {
                if (!runtime)
                    throw std::runtime_error(
                        "CUDA tensor execution requires libcuda at runtime");

                check_cuda(runtime.api, runtime.api.init(0), "cuInit");

                cuda_device selected_device = 0;
                check_cuda(runtime.api,
                           runtime.api.device_get(&selected_device, 0),
                           "cuDeviceGet");
                check_cuda(runtime.api,
                           runtime.api.ctx_create(&context, 0, selected_device),
                           "cuCtxCreate");
                check_cuda(runtime.api, runtime.api.stream_create(&stream, 0),
                           "cuStreamCreate");
            }

            ~cuda_environment() {
                if (stream != nullptr) (void)runtime.api.stream_destroy(stream);
            }

            cuda_runtime runtime;
            cuda_context context = nullptr;
            cuda_stream stream = nullptr;
            std::vector<std::shared_ptr<module_handle>> live_modules;
        };

        [[nodiscard]] cuda_environment& environment() {
            static cuda_environment* instance = new cuda_environment();
            check_cuda(instance->runtime.api,
                       instance->runtime.api.ctx_set_current(instance->context),
                       "cuCtxSetCurrent");
            return *instance;
        }

    } // namespace

    struct device_buffer_state {
        explicit device_buffer_state(size_t requested_bytes)
            : bytes(requested_bytes) {
            if (bytes == 0) return;

            cuda_environment& env = environment();
            check_cuda(env.runtime.api, env.runtime.api.mem_alloc(&ptr, bytes),
                       "cuMemAlloc");
        }

        device_buffer_state(const device_buffer_state&) = delete;
        device_buffer_state& operator=(const device_buffer_state&) = delete;

        ~device_buffer_state() {
            if (ptr == 0) return;

            try {
                cuda_environment& env = environment();
                (void)env.runtime.api.mem_free(ptr);
            }
            catch (...) {
            }
        }

        cuda_device_ptr ptr = 0;
        size_t bytes = 0;
    };

    namespace {

        void require_device_bytes(const device_buffer& buffer, size_t bytes,
                                  std::string_view name) {
            if (buffer.size_bytes() < bytes) {
                std::ostringstream message;
                message << name << " device buffer is too small";
                throw std::invalid_argument(message.str());
            }
        }

        void synchronize(cuda_environment& env) {
            check_cuda(env.runtime.api,
                       env.runtime.api.stream_synchronize(env.stream),
                       "cuStreamSynchronize");
            env.live_modules.clear();
        }

    } // namespace

    device_buffer::device_buffer(size_t bytes) {
        if (bytes != 0) state_ = std::make_shared<device_buffer_state>(bytes);
    }

    bool device_buffer::empty() const noexcept {
        return state_ == nullptr || state_->bytes == 0;
    }

    size_t device_buffer::size_bytes() const noexcept {
        return state_ == nullptr ? 0 : state_->bytes;
    }

    compilation_error::compilation_error(std::string log)
        : std::runtime_error(log), log_(std::move(log)) {}

    const std::string& compilation_error::log() const noexcept { return log_; }

    bool is_available() noexcept {
        if (!cuda_driver_available() || !static_cast<bool>(load_nvrtc()))
            return false;

        try {
            (void)environment();
            return true;
        }
        catch (...) {
            return false;
        }
    }

    std::string compile_to_ptx(std::string_view source,
                               const std::vector<std::string>& options) {
        const std::string key = cache_key(source, options);
        {
            std::scoped_lock lock{ptx_cache_mutex()};
            const auto cached = ptx_cache().find(key);
            if (cached != ptx_cache().end()) return cached->second;
        }

        nvrtc_runtime runtime = load_nvrtc();
        if (!runtime)
            throw std::runtime_error("CUDA JIT requires NVRTC at runtime");

        nvrtc_program raw_program = nullptr;
        const std::string source_text{source};
        check_nvrtc(
            runtime.api,
            runtime.api.create_program(&raw_program, source_text.c_str(),
                                       "porch_kernel.cu", 0, nullptr, nullptr),
            "nvrtcCreateProgram");
        program_handle program{raw_program, runtime.api};

        std::vector<const char*> option_ptrs;
        option_ptrs.reserve(options.size());
        for (const std::string& option : options)
            option_ptrs.push_back(option.c_str());

        const nvrtc_result compile_result = runtime.api.compile_program(
            program.get(), static_cast<int>(option_ptrs.size()),
            option_ptrs.empty() ? nullptr : option_ptrs.data());
        if (compile_result != nvrtc_success)
            throw compilation_error(program_log(runtime.api, program.get()));

        size_t ptx_size = 0;
        check_nvrtc(runtime.api,
                    runtime.api.get_ptx_size(program.get(), &ptx_size),
                    "nvrtcGetPTXSize");

        std::string ptx(ptx_size, '\0');
        check_nvrtc(runtime.api, runtime.api.get_ptx(program.get(), ptx.data()),
                    "nvrtcGetPTX");
        if (!ptx.empty() && ptx.back() == '\0') ptx.pop_back();

        {
            std::scoped_lock lock{ptx_cache_mutex()};
            ptx_cache().emplace(key, ptx);
        }
        return ptx;
    }

    device_buffer make_device_buffer(std::span<const float32_t> source) {
        device_buffer buffer{source.size_bytes()};
        copy_to_device(buffer, source);
        return buffer;
    }

    void copy_to_device(device_buffer& buffer,
                        std::span<const float32_t> source) {
        if (source.empty()) return;
        require_device_bytes(buffer, source.size_bytes(), "target");

        cuda_environment& env = environment();
        check_cuda(env.runtime.api,
                   env.runtime.api.memcpy_htod(
                       buffer.state_->ptr, source.data(), source.size_bytes()),
                   "cuMemcpyHtoD");
    }

    void copy_to_host(const device_buffer& buffer,
                      std::span<float32_t> target) {
        if (target.empty()) return;
        require_device_bytes(buffer, target.size_bytes(), "source");

        cuda_environment& env = environment();
        synchronize(env);
        check_cuda(env.runtime.api,
                   env.runtime.api.memcpy_dtoh(
                       target.data(), buffer.state_->ptr, target.size_bytes()),
                   "cuMemcpyDtoH");
    }

    void synchronize() { synchronize(environment()); }

    void launch_elementwise_binary(std::string_view ptx,
                                   std::string_view function,
                                   std::span<const float32_t> lhs,
                                   std::span<const float32_t> rhs,
                                   std::span<float32_t> out) {
        if (lhs.size() != rhs.size() || lhs.size() != out.size())
            throw std::invalid_argument("CUDA elementwise operands must match");

        device_buffer device_lhs = make_device_buffer(lhs);
        device_buffer device_rhs = make_device_buffer(rhs);
        device_buffer device_out{out.size_bytes()};
        launch_elementwise_binary(ptx, function, device_lhs, device_rhs,
                                  device_out, out.size());
        copy_to_host(device_out, out);
    }

    void launch_elementwise_binary(std::string_view ptx,
                                   std::string_view function,
                                   const device_buffer& lhs,
                                   const device_buffer& rhs, device_buffer& out,
                                   size_t count) {
        std::vector<const device_buffer*> inputs{&lhs, &rhs};
        launch_fused_elementwise(ptx, function, inputs, out, count);
    }

    void launch_fused_elementwise(
        std::string_view ptx, std::string_view function,
        const std::vector<const device_buffer*>& inputs, device_buffer& out,
        size_t count) {
        if (count == 0) return;

        const size_t bytes = count * sizeof(float32_t);
        for (const device_buffer* input : inputs) {
            if (input == nullptr) {
                throw std::invalid_argument("CUDA elementwise input is null");
            }
            require_device_bytes(*input, bytes, "input");
        }
        require_device_bytes(out, bytes, "out");

        cuda_environment& env = environment();
        const std::string ptx_text{ptx};
        cuda_module raw_module = nullptr;
        check_cuda(
            env.runtime.api,
            env.runtime.api.module_load_data(&raw_module, ptx_text.c_str()),
            "cuModuleLoadData");
        auto module =
            std::make_shared<module_handle>(raw_module, env.runtime.api);
        env.live_modules.push_back(module);

        cuda_function kernel = nullptr;
        const std::string function_name{function};
        check_cuda(env.runtime.api,
                   env.runtime.api.module_get_function(&kernel, module->get(),
                                                       function_name.c_str()),
                   "cuModuleGetFunction");

        const uint32_t block_size = 256;
        const uint32_t grid_size =
            static_cast<uint32_t>((count + block_size - 1) / block_size);
        uint64_t element_count = static_cast<uint64_t>(count);
        std::vector<cuda_device_ptr> input_ptrs;
        input_ptrs.reserve(inputs.size());
        for (const device_buffer* input : inputs) {
            input_ptrs.push_back(input->state_->ptr);
        }
        cuda_device_ptr out_ptr = out.state_->ptr;
        std::vector<void*> params;
        params.reserve(input_ptrs.size() + 2);
        for (cuda_device_ptr& input_ptr : input_ptrs) {
            params.push_back(&input_ptr);
        }
        params.push_back(&out_ptr);
        params.push_back(&element_count);

        check_cuda(env.runtime.api,
                   env.runtime.api.launch_kernel(
                       kernel, grid_size, 1, 1, block_size, 1, 1, 0, env.stream,
                       params.data(), nullptr),
                   "cuLaunchKernel");
    }

} // namespace porch::cuda_jit
