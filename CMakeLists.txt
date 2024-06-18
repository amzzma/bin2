cmake_minimum_required(VERSION 3.24)  # This my version, and you can alter.
project(bin2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")

set(Include_path "${CMAKE_CURRENT_SOURCE_DIR}/Kernel/include")

function(AddFiles _out)
    file(GLOB_RECURSE _tmp ${ARGN})
    list(APPEND ${_out} ${_tmp})
    set(${_out} ${${_out}} PARENT_SCOPE)
endfunction()

set(SOURCES)
AddFiles(SOURCES
    "${PROJECT_SOURCE_DIR}/Kernel/include/*.h"
    "${PROJECT_SOURCE_DIR}/Kernel/src/*.cpp"
)

message("SOURCES: ${SOURCES}")

add_library(kernel STATIC ${SOURCES})
target_include_directories(kernel PRIVATE ${Include_path})

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(kernel PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bin2_demo "test_bin.cpp" "main.cpp")
target_include_directories(bin2_demo PRIVATE ${Include_path})
target_link_libraries(bin2_demo PRIVATE kernel)

