cmake_minimum_required(VERSION 3.16)
project(hip_functions LANGUAGES CXX)

# Find HIP
find_package(hip REQUIRED)

# Set HIP compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Link HIP runtime
target_link_libraries(hip_functions PUBLIC hip::host)

# Set include directories
target_include_directories(hip_functions PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Set compiler for HIP files
set_target_properties(hip_functions PROPERTIES 
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
)

# Create executable for testing (optional)
add_executable(wa1-task3 wa1-task3-hipified.cu)
set_source_files_properties(wa1-task3-hipified.cu PROPERTIES LANGUAGE HIP)
target_link_libraries(wa1-task3 PUBLIC hip::host)