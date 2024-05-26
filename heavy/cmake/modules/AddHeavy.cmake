include(AddLLVM)
include(LLVMDistributionSupport)

function(add_heavy_library name)
set(options "")
set(one_value_args "")
set(multi_value_args "")
cmake_parse_arguments(ARG "${options}" "${one_value_args}"
                      "${multi_value_args}" "${ARGN}")
llvm_add_library(${name} ${ARG_UNPARSED_ARGUMENTS})
get_target_export_arg(${name} Heavy export_arg UMBRELLA heavy-libraries)
install(TARGETS ${name}
        COMPONENT ${name}
        ${export_arg})

add_llvm_install_targets(install-${name}
                         DEPENDS ${name}
                         COMPONENT ${name})
set_property(GLOBAL APPEND PROPERTY HEAVY_EXPORTS ${name})
endfunction(add_heavy_library)

# add_heavy_scheme_module
# - Create a scheme module that is, by default, dynamically loaded.
# - TODO Implement this.
function(add_heavy_scheme_module name)
set(options STATIC)
set(one_value_args "")
set(multi_value_args "")
cmake_parse_arguments(ARG "${options}" "${one_value_args}"
                      "${multi_value_args}")
endfunction(add_heavy_scheme_module name)
