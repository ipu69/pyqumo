include_directories(../core)

add_library(
        cqumo_oqnet
        STATIC
        cqumo/oqnet/base.h
        cqumo/oqnet/base.cpp
        cqumo/oqnet/components.h
        cqumo/oqnet/components.cpp
        cqumo/oqnet/journals.h
        cqumo/oqnet/journals.cpp
        cqumo/oqnet/marshal.h
        cqumo/oqnet/marshal.cpp
        cqumo/oqnet/simulation.h
        cqumo/oqnet/simulation.cpp
        cqumo/oqnet/statistics.h
        cqumo/oqnet/statistics.cpp
        cqumo/oqnet/system.h
        cqumo/oqnet/system.cpp
)

add_executable(
        oqnet
        main.cpp
)

target_link_libraries(
        oqnet
        cqumo_oqnet
        cqumo_core
)

#find_package(GTest)
#if(GTEST_FOUND)
#    add_executable(
#            test_cqumo_oqnet
#            tests/test_foo.cpp
#    )
#    target_include_directories(
#            test_cqumo_oqnet
#            PRIVATE
#            ./
#    )
#    target_link_libraries(
#            test_cqumo_oqnet
#            cqumo_oqnet
#            cqumo_core
#            GTest::gmock
#            GTest::gmock_main
#    )
#endif()
