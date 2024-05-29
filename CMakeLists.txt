cmake_minimum_required(VERSION 3.15..3.26)

find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

project(neural_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})
message("Compiler: " ${CMAKE_CXX_COMPILER})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all -pthread")

# Ignore annoying Intel warnings
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=2196")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

# -------------------------------- #
# ------ Fetching externals ------ #
# -------------------------------- #

include(FetchContent)

set(EXTERNALS "")
if (DEFINED UTILS_SOURCE)
    set(utils_SOURCE_DIR ${UTILS_SOURCE})
    add_subdirectory(${UTILS_SOURCE} "utils_src")
else()
    FetchContent_Declare(
        utils
        GIT_REPOSITORY https://github.com/eliotheinrich/utils.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    list(APPEND EXTERNALS utils)
endif()

if (DEFINED DATAFRAME_SOURCE)
    set(dataframe_SOURCE_DIR ${DATAFRAME_SOURCE})
    add_subdirectory(${DATAFRAME_SOURCE} "dataframe_src")
else()
    FetchContent_Declare(
        dataframe
        GIT_REPOSITORY https://github.com/eliotheinrich/dataframe
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    list(APPEND EXTERNALS dataframe)
endif()

#FetchContent_Declare(
#    eigen
#    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
#    GIT_TAG master
#    GIT_SHALLOW TRUE
#    GIT_PROGRESS TRUE
#)
#set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
#set(EIGEN_BUILD_DOC OFF)
#set(BUILD_TESTING OFF)
#set(EIGEN_BUILD_PKGCONFIG OFF)
#list(APPEND EXTERNALS eigen)

FetchContent_MakeAvailable(${EXTERNALS})

include_directories(
    src
    ${dataframe_SOURCE_DIR}/src
    ${utils_SOURCE_DIR}/src
)

# Adding Python bindings
find_package(Python 3.10 
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    neural_network_bindings 
    STABLE_ABI
    NB_STATIC
    src/NoisyNeuralSimulator.hpp
    src/PyNeural.cpp
)

set_target_properties(neural_network_bindings PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(neural_network_bindings PRIVATE src)
target_link_libraries(neural_network_bindings PRIVATE dataframe)

install(
    TARGETS neural_network_bindings
    LIBRARY DESTINATION pyneural
)
