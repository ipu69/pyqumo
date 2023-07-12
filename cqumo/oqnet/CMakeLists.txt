include_directories(
        ../core/include
        ./include
)

add_library(
        cqumo_oqnet
        STATIC
        include/cqumo/oqnet.h
        src/cqumo/oqnet/components.h
        src/cqumo/oqnet/components.cpp
        src/cqumo/oqnet/journals.h
        src/cqumo/oqnet/journals.cpp
        src/cqumo/oqnet/marshal.h
        src/cqumo/oqnet/marshal.cpp
        src/cqumo/oqnet/simulation.h
        src/cqumo/oqnet/simulation.cpp
        src/cqumo/oqnet/system.h
        src/cqumo/oqnet/system.cpp
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
