project (test)

# unit tests
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.15.2
  VERSION 1.15.2
  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

set(GTEST_TESTS
    unittest_masstree
    unittest_linearhashtable
    unittest_fixedhashtable
)
foreach(TEST ${GTEST_TESTS})
    add_executable(${TEST} ${TEST}.cu)
    set_target_properties(${TEST} PROPERTIES FOLDER "tests")
    target_link_libraries(${TEST} PRIVATE robust_gpu_indexing PRIVATE gtest)
endforeach()


