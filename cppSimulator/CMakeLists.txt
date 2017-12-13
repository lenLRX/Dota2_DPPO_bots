cmake_minimum_required (VERSION 3.1)
project(cppSimulator)

set(CMAKE_CXX_STANDARD 11)


if (WIN32)
    set(PYTHON_LIBRARIES "C:\\Users\\abceq\\Anaconda3\\libs")
    set(PYTHON_INCLUDE_DIRS "C:\\Users\\abceq\\Anaconda3\\include")
endif (WIN32)

if (UNIX)
    FIND_PACKAGE(PythonLibs)
    if (PYTHONLIBS_FOUND)
        MESSAGE("Python LIB: " ${PYTHON_LIBRARIES})
        MESSAGE("Python INC: " ${PYTHON_INCLUDE_DIRS})
        link_libraries(${PYTHON_LIBRARIES})
    else (NOT PYTHONLIBS_FOUND)
        set(PYTHON_LIBRARIES "/root/anaconda3/lib")
        set(PYTHON_INCLUDE_DIRS "/root/anaconda3/include/python3.6m")
        link_libraries(python3.6m)
    endif (PYTHONLIBS_FOUND)
endif(UNIX)

link_directories(${PYTHON_LIBRARIES})
include_directories(${PYTHON_INCLUDE_DIRS})

include_directories(include)

file(GLOB HEADERS include/*.h)

aux_source_directory(src SRCS)
add_library(cppSimulator SHARED ${SRCS} ${HEADERS})


add_executable(testSimulator ${SRCS} ${HEADERS} ./test/test_main.cpp)

set_target_properties(testSimulator
    PROPERTIES 
    EXECUTABLE_OUTPUT_PATH ..
)

if (WIN32)
    set(LIB_SUFFIX ".pyd")
endif (WIN32)

if (UNIX)
    set(LIB_SUFFIX ".so")
endif(UNIX)

set_target_properties(cppSimulator
    PROPERTIES 
    PREFIX ""
    SUFFIX ${LIB_SUFFIX}
    LIBRARY_OUTPUT_DIRECTORY ..
)

if (WIN32)
    add_custom_command(TARGET cppSimulator POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE_DIR:cppSimulator>/cppSimulator${LIB_SUFFIX}"
    "${CMAKE_SOURCE_DIR}/cppSimulator${LIB_SUFFIX}"              
    )
endif (WIN32)
