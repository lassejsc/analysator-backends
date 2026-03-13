cmake_minimum_required(VERSION 3.10)
project(cpphelpers)

# Find Python interpreter + development headers + extension suffix
find_package(Python3 $ENV{PY_VERSION} EXACT COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Build the extension module
add_library(cpphelpers SHARED src/test.cpp)


add_library(vlsvrs SHARED IMPORTED)
set_property(TARGET vlsvrs PROPERTY IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../target/debug/libvlsvrs.so")

# link_directories(${CMAKE_BINARY_DIR})
# message("TEST" ${CMAKE_BINARY_DIR})
# Link Python
target_link_libraries(cpphelpers PRIVATE Python3::Module vlsvrs Python3::NumPy)
# Ensure correct naming (no "lib" prefix, correct .so/.pyd suffix)
set_target_properties(cpphelpers PROPERTIES
    PREFIX ""                                  # remove "lib"
    SUFFIX ".so"       # e.g. .so, .pyd
)
install(TARGETS cpphelpers LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../)
