cmake_minimum_required(VERSION 3.10)
project(cpphelpers)

# Find Python interpreter + development headers + extension suffix
find_package(Python3 $ENV{PY_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)

# Build the extension module
add_library(cpphelpers SHARED src/cpphelpers.cpp)

# Link Python
target_link_libraries(cpphelpers PRIVATE Python3::Module)


# Ensure correct naming (no "lib" prefix, correct .so/.pyd suffix)
set_target_properties(cpphelpers PROPERTIES
    PREFIX ""                                  # remove "lib"
    SUFFIX ".so"       # e.g. .so, .pyd
)
install(TARGETS cpphelpers LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../)
