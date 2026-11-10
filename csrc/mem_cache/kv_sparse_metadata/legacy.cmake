# Add only this Torch adapter to the legacy library. CANN launches the
# device kernel from the packaged vendor built by the new framework.
set(kv_sparse_metadata_adapter "${CMAKE_CURRENT_LIST_DIR}/torch_adapter.cpp")
target_sources(${OP_PLUGIN_NAME} PRIVATE "${kv_sparse_metadata_adapter}")
set_property(SOURCE "${kv_sparse_metadata_adapter}" APPEND PROPERTY INCLUDE_DIRECTORIES
    "${PROJECT_OP_SRC_BASE}/aclnn_torch_adapter"
    "${CMAKE_CURRENT_LIST_DIR}/op_host"
    "${ASCEND_HOME_PATH}/include")
find_library(SGL_KV_SPARSE_METADATA_NNOPBASE nnopbase
    PATHS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(${OP_PLUGIN_NAME} PRIVATE "${SGL_KV_SPARSE_METADATA_NNOPBASE}")

if(SGL_KERNEL_ARCH STREQUAL "arch35")
    set(kv_sparse_metadata_arch 3510)
else()
    set(kv_sparse_metadata_arch 2201)
endif()
get_filename_component(kv_sparse_metadata_package_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)
add_custom_target(sgl_legacy_kv_sparse_metadata_vendor
    COMMAND "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
        "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/stage_legacy_vendor.py"
        "${kv_sparse_metadata_arch}" "${CMAKE_CURRENT_BINARY_DIR}/kv_sparse_metadata"
        "${kv_sparse_metadata_package_dir}/_cann/vendors/sgl_kernel_npu"
    VERBATIM)
add_dependencies(${OP_PLUGIN_NAME} sgl_legacy_kv_sparse_metadata_vendor)
