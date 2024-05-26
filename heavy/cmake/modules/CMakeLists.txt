set(HEAVY_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/heavy" CACHE STRING
  "Path for CMake subdirectory for Heavy (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/heavy')")
set(heavy_cmake_builddir "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/heavy")
get_property(HEAVY_EXPORTS GLOBAL PROPERTY HEAVY_EXPORTS)
export(TARGETS ${HEAVY_EXPORTS} FILE ${heavy_cmake_builddir}/HeavyTargets.cmake)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install_distribution_exports(Heavy)
endif()
