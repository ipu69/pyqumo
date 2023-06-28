add_library(
        cqumo_core
        STATIC
        cqumo/functions.h
        cqumo/functions.cpp
        cqumo/randoms/randoms.h
        cqumo/randoms/randoms.cpp
)

find_package(GTest)
if(GTEST_FOUND)
    add_executable(
        test_cqumo_core
            tests/test_functions.cpp
    )
    target_include_directories(test_cqumo_core PRIVATE ./)
    target_link_libraries(
        test_cqumo_core
        cqumo_core
        GTest::gmock 
        GTest::gmock_main
    )
endif()
