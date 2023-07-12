add_library(
        cqumo_core
        STATIC
        include/cqumo/cqumo.h
        src/cqumo/functions.h
        src/cqumo/functions.cpp
        src/cqumo/randoms.h
        src/cqumo/randoms.cpp
        src/cqumo/statistics.h
        src/cqumo/statistics.cpp
        src/cqumo/strings.h
)

find_package(GTest)
if(GTEST_FOUND)
    add_executable(
        test_cqumo_core
            tests/test_functions.cpp
    )
    target_include_directories(test_cqumo_core PRIVATE ./src)
    target_link_libraries(
        test_cqumo_core
        cqumo_core
        GTest::gmock 
        GTest::gmock_main
    )
endif()
