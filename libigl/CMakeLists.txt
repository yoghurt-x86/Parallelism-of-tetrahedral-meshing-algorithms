cmake_minimum_required(VERSION 3.16)
project(example)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)
# Other modules you could enable
#igl_include(embree)
igl_include(imgui)
#igl_include(opengl)
igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
#igl_include(copyleft core)
#igl_include(copyleft cork)
igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
#igl_include(restricted triangle)

# Fetch CGAL from GitHub
include(FetchContent)
FetchContent_Declare(
  CGAL
  GIT_REPOSITORY https://github.com/CGAL/cgal.git
  GIT_TAG master
)
FetchContent_MakeAvailable(CGAL)

# Set CGAL_DIR to the fetched CGAL location
set(CGAL_DIR "${cgal_SOURCE_DIR}")

# Find CGAL package
find_package(CGAL REQUIRED)

# Find HIP for GPU acceleration
find_package(hip QUIET)
if(hip_FOUND)
    message(STATUS "HIP found, enabling GPU acceleration")
    set(HIP_ENABLED ON)
    
    # Set Eigen include directory for hip subdirectory
    get_target_property(EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS} PARENT_SCOPE)
    
    # Add hip subdirectory
    add_subdirectory(../hip hip_build)
else()
    message(STATUS "HIP not found, GPU acceleration disabled")
    set(HIP_ENABLED OFF)
endif()

# Shared library for common functionality
add_library(tetmesh_lib TetMesh.cpp)

# Main UI application
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tetmesh_lib)

# Second program (you can customize this)
add_executable(tetmesh_processor processor_main.cpp)
target_link_libraries(tetmesh_processor PRIVATE tetmesh_lib)

# Enable modules for GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -fmodules-ts)
    target_compile_options(tetmesh_processor PRIVATE -fmodules-ts)
endif()

# Enable modules for Clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -fmodules)
    target_compile_options(tetmesh_processor PRIVATE -fmodules)
endif()

# Link libraries to shared tetmesh_lib
target_link_libraries(tetmesh_lib PUBLIC 
  igl::glfw
  ## Other modules you could link to
  # igl::embree
  igl::imgui
  # igl::opengl
  igl::stb
  # igl::predicates
  # igl::xml
  #igl_copyleft::cgal
  # igl_copyleft::comiso
  # igl_copyleft::core
  # igl_copyleft::cork
  igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
  CGAL::CGAL
  )

# Link igl (and the glfw module) to your main UI project
target_link_libraries(${PROJECT_NAME} PUBLIC 
  igl::glfw
  igl::imgui
  igl::stb
  igl_copyleft::tetgen
  CGAL::CGAL
  )

# Link required libraries to processor (no UI libraries needed)
target_link_libraries(tetmesh_processor PUBLIC 
  igl_copyleft::tetgen
  CGAL::CGAL
  )

# Link HIP functions if available
if(HIP_ENABLED)
    target_link_libraries(tetmesh_lib PUBLIC hip_functions)
    target_compile_definitions(tetmesh_lib PRIVATE HIP_ENABLED)
    target_link_libraries(${PROJECT_NAME} PUBLIC hip_functions)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HIP_ENABLED)
    target_link_libraries(tetmesh_processor PUBLIC hip_functions)
    target_compile_definitions(tetmesh_processor PRIVATE HIP_ENABLED)
endif()
