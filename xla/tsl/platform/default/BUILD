# Tensorflow default + linux implementations of tensorflow/core/platform libraries.
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load(
    "//xla/tsl:tsl.bzl",
    "if_not_fuchsia",
    "if_not_windows",
    "if_oss",
    "internal_visibility",
    "tsl_copts",
)
load(
    "//xla/tsl:tsl.default.bzl",
    "filegroup",
    "if_cuda_tools",
    "tsl_grpc_cc_dependencies",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@tsl//tsl/platform:__pkg__",
        "//xla/tsl/platform:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
)

cc_library(
    name = "context",
    hdrs = ["@tsl//tsl/platform:context.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["context.h"],
    deps = [
        "@tsl//tsl/platform",
    ],
)

cc_library(
    name = "criticality",
    hdrs = ["//xla/tsl/platform:criticality.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    textual_hdrs = ["criticality.h"],
    deps = [
        "@tsl//tsl/platform",
    ],
)

cc_library(
    name = "cuda_root_path",
    srcs = ["cuda_root_path.cc"],
    hdrs = ["@tsl//tsl/platform:cuda_root_path.h"],
    compatible_with = [],
    data = if_cuda_tools([
        "@cuda_nvcc//:nvvm",
    ]),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "dso_loader",
    srcs = ["dso_loader.cc"] + select({
        # include dynamic loading checker only for open source build
        "//xla/tsl:oss": ["dlopen_checker.cc"],
        "//conditions:default": ["dlopen_checker_stub.cc"],
    }),
    hdrs = ["dso_loader.h"],
    compatible_with = [],
    copts = tsl_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:tensorrt_headers",
        "@tsl//tsl/platform:load_library",
        "@tsl//tsl/platform:path",
    ] + if_oss([
        "@nvshmem//:nvshmem_config",
        "@local_config_nccl//:nccl_config",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_config",
        "@local_config_rocm//rocm:rocm_headers",
    ]),
)

cc_library(
    name = "env",
    srcs = [
        "posix_file_system.cc",
        "//xla/tsl/platform:env.cc",
        "//xla/tsl/platform:file_system.cc",
        "//xla/tsl/platform:file_system_helper.cc",
        "//xla/tsl/platform:threadpool.cc",
    ],
    hdrs = [
        "posix_file_system.h",
        "//xla/tsl/platform:env.h",
        "//xla/tsl/platform:file_system.h",
        "//xla/tsl/platform:file_system_helper.h",
        "//xla/tsl/platform:ram_file_system.h",
        "//xla/tsl/platform:threadpool.h",
    ],
    copts = tsl_copts(),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:file_statistics",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:threadpool_interface",
        "//xla/tsl/platform:types",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:blocking_counter",
        "@tsl//tsl/platform:context",
        "@tsl//tsl/platform:cord",
        "@tsl//tsl/platform:denormal",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:regexp",
        "@tsl//tsl/platform:scanner",
        "@tsl//tsl/platform:setround",
        "@tsl//tsl/platform:str_util",
        "@tsl//tsl/platform:strcat",
        "@tsl//tsl/platform:stringpiece",
        "@tsl//tsl/platform:stringprintf",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:tracing",
    ],
)

cc_library(
    name = "env_impl",
    srcs = [
        "env.cc",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:load_library",
        "@tsl//tsl/platform:strcat",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "env_time",
    srcs = ["env_time.cc"],
    hdrs = ["//xla/tsl/platform:env_time.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["//xla/tsl/platform:types"],
)

cc_library(
    name = "error_logging",
    srcs = ["error_logging.cc"],
    hdrs = ["@tsl//tsl/platform:error_logging.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grpc_credentials",
    srcs = ["grpc_credentials.cc"],
    hdrs = ["//xla/tsl/platform:grpc_credentials.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/log:check",
    ] + tsl_grpc_cc_dependencies(),
)

cc_library(
    name = "human_readable_json",
    srcs = ["human_readable_json.cc"],
    hdrs = ["@tsl//tsl/platform:human_readable_json.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:protobuf",
        "@tsl//tsl/platform:strcat",
        "@tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "load_library",
    srcs = ["load_library.cc"],
    hdrs = ["@tsl//tsl/platform:load_library.h"],
    linkstatic = True,
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ] + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_rpath",
    ]),
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["//xla/tsl/platform:logging.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["logging.h"],
    deps = [
        "//xla/tsl/platform:env_time",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform",
    ],
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "context.h",
        "integral_types.h",
    ] + if_not_windows(["env_time.cc"]),
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "@tsl//tsl/platform:net.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "@tsl//tsl/platform:strcat",
    ],
    alwayslink = True,
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "@tsl//tsl/platform:cpu_info.cc",
    ],
    hdrs = [
        "//xla/tsl/platform/profile_utils:cpu_utils.h",
        "@tsl//tsl/platform:cpu_info.h",
        "@tsl//tsl/platform:demangle.h",
        "@tsl//tsl/platform:host_info.h",
        "@tsl//tsl/platform:init_main.h",
        "@tsl//tsl/platform:mem.h",
        "@tsl//tsl/platform:numa.h",
        "@tsl//tsl/platform:snappy.h",
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"] + select({
        # TF Additional NUMA defines
        "//xla/tsl:with_numa_support": ["TENSORFLOW_USE_NUMA"],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:byte_order",
        "//xla/tsl/platform:dynamic_annotations",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "//xla/tsl/platform/profile_utils:profile_utils_cpu_utils",
        "@com_google_absl//absl/base",
        "@snappy",
        "@tsl//tsl/platform",
    ] + select({
        # TF Additional NUMA dependencies
        "//xla/tsl:with_numa_support": [
            # Don't merge in a single line
            "@hwloc",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "resource",
    srcs = ["resource.cc"],
    hdrs = ["//xla/tsl/platform:resource.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "rocm_rocdl_path",
    srcs = ["rocm_rocdl_path.cc"],
    hdrs = ["//xla/tsl/platform:rocm_rocdl_path.h"],
    compatible_with = [],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "@local_config_rocm//rocm:rocm_config",
        "@local_config_rocm//rocm:rocm_headers",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "stacktrace",
    hdrs = ["stacktrace.h"],
    linkopts = ["-ldl"],
    deps = [
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:abi",
    ],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["@tsl//tsl/platform:stacktrace_handler_hdrs"],
    linkstatic = 1,
    deps = [
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:stacktrace",
    ],
    alwayslink = 1,
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["//xla/tsl/platform:subprocess.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["subprocess.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:thread_annotations",
    ],
    alwayslink = True,
)

cc_library(
    name = "tracing",
    srcs = [
        "tracing.cc",
        "@tsl//tsl/platform:tracing.cc",
    ],
    hdrs = ["@tsl//tsl/platform:tracing.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["tracing_impl.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/backends/cpu:threadpool_listener_state",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:hash",
        "@tsl//tsl/platform:str_util",
        "@tsl//tsl/platform:strcat",
        "@tsl//tsl/platform:stringpiece",
    ],
    alwayslink = True,
)

cc_library(
    name = "types",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["integral_types.h"],
)

cc_library(
    name = "unbounded_work_queue",
    srcs = ["unbounded_work_queue.cc"],
    hdrs = ["unbounded_work_queue.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "crash_analysis",
    srcs = [
        "crash_analysis.cc",
    ],
    hdrs = ["//xla/tsl/platform:crash_analysis.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["crash_analysis.h"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
    deps = [
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "status",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["status.h"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
)

cc_library(
    name = "statusor",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["statusor.h"],
    visibility = internal_visibility(["//tensorflow:__subpackages__"]),
    deps = [
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/status:statusor",
    ],
)

bzl_library(
    name = "cuda_build_defs_bzl",
    srcs = ["cuda_build_defs.bzl"],
    # copybara:uncomment_begin
    # test_tags = [
    # "manual",
    # "notap",
    # ],
    # visibility = internal_visibility(["//tensorflow:__subpackages__"]),
    # copybara:uncomment_end
    deps = [
        "//xla/tsl:package_groups_bzl",
    ],
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
    # This .bzl file is only used in OSS and references OSS paths. Therefore it cannot be parsed
    # correctly by the parse test in g3.
    # copybara:uncomment parse_tests = False,
)

bzl_library(
    name = "build_config_root_bzl",
    srcs = ["build_config_root.bzl"],
    # copybara:uncomment parse_tests = False,
    deps = [
        "//xla/tsl:package_groups_bzl",
        "@xla//third_party/py/rules_pywrap:pywrap_bzl",
    ],
)

bzl_library(
    name = "build_config_bzl",
    srcs = ["build_config.bzl"],
    # copybara:uncomment parse_tests = False,
    deps = [
        ":build_config_root_bzl",
        "//xla/tsl:tsl_bzl",
        "@xla//third_party/py/rules_pywrap:pywrap_bzl",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "additional_mobile_srcs_no_runtime",
    visibility = internal_visibility(["//tensorflow/core/platform:__pkg__"]),
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "casts.h",
        "context.h",
        "env.cc",
        "integral_types.h",
        "load_library.cc",
        "port.cc",
        "posix_file_system.cc",
        "posix_file_system.h",
        "stacktrace.h",
        "status.h",
        "statusor.h",
        "tracing_impl.h",
        "//xla/tsl/platform/profile_utils:cpu_utils.h",
        "//xla/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    visibility = internal_visibility([
        "//tensorflow/core/platform:__pkg__",
        "@tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "casts.h",
        "error_logging.cc",
        "unbounded_work_queue.cc",
        "unbounded_work_queue.h",
    ] + if_not_fuchsia([
        "subprocess.cc",
        "subprocess.h",
    ]),
    visibility = internal_visibility(["//tensorflow/core/platform:__pkg__"]),
)

exports_files(
    srcs = glob(
        ["*"],
        exclude = [
            "integral_types.h",
            "logging.h",
            "test.cc",
        ],
    ),
    visibility = internal_visibility(["//tensorflow/core/platform:__pkg__"]),
)

exports_files(
    srcs = [
        "integral_types.h",
        "logging.h",
        "test.cc",
    ],
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

cc_library(
    name = "recordphase",
    srcs = ["recordphase.cc"],
    hdrs = ["//xla/tsl/platform:recordphase.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)
