add_library(cqumo STATIC "")
add_subdirectory(cqumo)

find_package(GTest)
if(GTEST_FOUND)
    add_executable(
            test_cqumo
            test/test_functions.cpp
    )
    target_include_directories(test_cqumo PRIVATE .)
    target_link_libraries(
            test_cqumo
            cqumo
            GTest::gmock
            GTest::gmock_main
    )
endif()
