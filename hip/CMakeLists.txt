cmake_minimum_required(VERSION 3.16)
project(hip_functions LANGUAGES CXX HIP)

# Find HIP
find_package(hip REQUIRED)

# Find rocPRIM (required by hipCUB)
find_package(rocprim REQUIRED)

# Find hipCUB
find_package(hipcub REQUIRED)

# Set HIP architecture targets (fallback to common ones)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx942")
endif()

# Set HIP compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Set HIP-specific compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add the HIP source files
set(HIP_SOURCES
    wa1-task3-hipified.cu
    vertex_processor.cu
)

# Create a library from HIP sources
add_library(hip_functions STATIC ${HIP_SOURCES})

# Set HIP compiler for the sources
set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

# Link HIP runtime and hipCUB
target_link_libraries(hip_functions PUBLIC hip::host hip::hipcub)

# Set include directories
target_include_directories(hip_functions PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add Eigen include directories if available
if(DEFINED EIGEN_INCLUDE_DIRS)
    target_include_directories(hip_functions PUBLIC ${EIGEN_INCLUDE_DIRS})
    message(STATUS "Added Eigen include directories to hip_functions: ${EIGEN_INCLUDE_DIRS}")
endif()

# Set compiler for HIP files
set_target_properties(hip_functions PROPERTIES 
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)

# Create executable for testing (optional)
add_executable(wa1-task3 wa1-task3-hipified.cu)
set_source_files_properties(wa1-task3-hipified.cu PROPERTIES LANGUAGE HIP)
target_link_libraries(wa1-task3 PUBLIC hip::host hip::hipcub)

# Add Eigen include directories to test executable if available
if(DEFINED EIGEN_INCLUDE_DIRS)
    target_include_directories(wa1-task3 PUBLIC ${EIGEN_INCLUDE_DIRS})
endif()