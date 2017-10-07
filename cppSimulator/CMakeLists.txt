cmake_minimum_required (VERSION 2.6)
project(cppSimulator)


if (WIN32)
    set(PYTHON_LIBRARY "C:\\Users\\abceq\\Anaconda3\\libs")
    set(PYTHON_INCLUDE_DIR "C:\\Users\\abceq\\Anaconda3\\include")
endif (WIN32)

if (LINUX)
    set(PYTHON_LIBRARY "/root/anaconda3/lib")
    set(PYTHON_INCLUDE_DIR "/root/anaconda3/include/python3.6m")
endif(LINUX)

FIND_PACKAGE(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARY})

include_directories(include)

file(GLOB HEADERS include/*.h)

aux_source_directory(src SRCS)
add_library(cppSimulator SHARED ${SRCS} ${HEADERS})

if (WIN32)
    set(LIB_SUFFIX ".pyd")
endif (WIN32)

if (LINUX)
    set(LIB_SUFFIX ".so")
endif(LINUX)

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