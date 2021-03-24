add_custom_target(HeavyUnitTests)
function(add_heavy_unittest test_name)
  add_unittest(HeavyUnitTests ${test_name} ${ARGN})
endfunction()

add_heavy_unittest(HeavyValueTest ValueTest.cpp)
