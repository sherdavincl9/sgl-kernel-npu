# Register both KV transfer adapters in the legacy Torch library. The ACLNN
# headers and device kernels come from the same packaged CANN vendor.
get_filename_component(kv_transfer_package_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)
set(kv_transfer_vendor "${kv_transfer_package_dir}/_cann/vendors/sgl_kernel_npu")
set(kv_transfer_adapters
    "${PROJECT_OP_SRC_BASE}/mem_cache/kv_exchange/torch_adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/torch_adapter.cpp")
target_sources(${OP_PLUGIN_NAME} PRIVATE ${kv_transfer_adapters})
set_property(SOURCE ${kv_transfer_adapters} APPEND PROPERTY INCLUDE_DIRECTORIES
    "${PROJECT_OP_SRC_BASE}/aclnn_torch_adapter"
    "${kv_transfer_vendor}/op_api/include"
    "${ASCEND_HOME_PATH}/include")
find_library(SGL_KV_TRANSFER_NNOPBASE nnopbase
    PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(${OP_PLUGIN_NAME} PRIVATE "${SGL_KV_TRANSFER_NNOPBASE}")

if(SGL_KERNEL_ARCH STREQUAL "arch35")
    set(kv_transfer_arch 3510)
else()
    set(kv_transfer_arch 2201)
endif()
add_custom_target(sgl_legacy_kv_transfer_vendor
    COMMAND "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
        "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/stage_legacy_vendor.py"
        "${kv_transfer_arch}" "${CMAKE_CURRENT_BINARY_DIR}/kv_transfer"
        "${kv_transfer_vendor}"
    VERBATIM)
add_dependencies(${OP_PLUGIN_NAME} sgl_legacy_kv_transfer_vendor)
