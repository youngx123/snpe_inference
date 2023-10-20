cmake_minimum_required(VERSION 3.10)
set(CMAKE_VERBOSE_MAKEFILE on)
PROJECT(snpe_infer_demo)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD 14)
# set(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

# set(PLATFORM qnx-aarch64le)
set(PLATFORM Linux-x86_64)
message("platform : "${PLATFORM})

set(snpe_path ${CMAKE_SOURCE_DIR}/3thirdParty/snpe/v1.65/${PLATFORM})
set(SNPE_INC ${snpe_path}/include)
set(SNPE_LIB ${snpe_path}/lib)


link_directories(
        ${SNPE_LIB}
)

# set(OpenCV_DIR ${Lib_PATH}/OpenCV/${PLATFORM})
set(opencv_path ${CMAKE_SOURCE_DIR}/3thirdParty/OpenCV/${PLATFORM})
set( ENV{OpenCV_DIR} ${opencv_path})

find_package(OpenCV REQUIRED)

message(STATUS "    OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${SNPE_INC}/zdl
        ${SNPE_INC}
        ${OpenCV_INCLUDE_DIRS}
)

add_executable( # Sets the name.
        ${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/snpe_infer/snpe_infer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/createBuffer.cpp
        )

target_link_libraries(
        ${PROJECT_NAME}
        SNPE
        ${OpenCV_LIBS}
)