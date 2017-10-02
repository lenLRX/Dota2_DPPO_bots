cmake_minimum_required (VERSION 2.6)
project(cppSimulator)

set(PYTHON_LIBRARY "C:\\Users\\abceq\\Anaconda3\\libs")
set(PYTHON_INCLUDE_DIR "C:\\Users\\abceq\\Anaconda3\\include")
FIND_PACKAGE(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARY})

aux_source_directory(src SRCS)
add_library(cppSimulator SHARED ${SRCS})



if (WIN32)
    set(LIB_SUFFIX ".pyd")
endif (WIN32)

set_target_properties(cppSimulator
    PROPERTIES 
    SUFFIX ${LIB_SUFFIX}
    LIBRARY_OUTPUT_DIRECTORY .
)

add_custom_command(TARGET cppSimulator POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:cppSimulator>/cppSimulator${LIB_SUFFIX}"
    "${CMAKE_SOURCE_DIR}/cppSimulator${LIB_SUFFIX}"              
)