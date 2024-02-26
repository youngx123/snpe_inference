cmake_minimum_required(VERSION 3.10)
set(CMAKE_VERBOSE_MAKEFILE on)
PROJECT(snpe_infer_demo)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD 14)

set(PLATFORM qnx-aarch64le)
#set(PLATFORM Linux-x86_64)
message("platform : "${PLATFORM})

set(snpe_path ${CMAKE_CURRENT_SOURCE_DIR}/3thirdParty/snpe/v1.51/${PLATFORM})
set(SNPE_INC ${snpe_path}/include)
set(SNPE_LIB ${snpe_path}/lib)


link_directories(
        ${SNPE_LIB}
)

set(opencv_path ${CMAKE_CURRENT_SOURCE_DIR}/3thirdParty/OpenCV/${PLATFORM})
set( ENV{OpenCV_DIR} ${opencv_path})

find_package(OpenCV REQUIRED)

message(STATUS "    OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/detect/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/snpe/
        ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/utils/
        ${SNPE_INC}/zdl
        ${SNPE_INC}  
        ${OpenCV_INCLUDE_DIRS}
)

add_executable( # Sets the name.
        ${PROJECT_NAME}
         ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/nn_inference.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/inference_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/utils/createBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/snpe/snpe_infer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/detect/face_landmark.cpp
    )

target_link_libraries(
        ${PROJECT_NAME}
        SNPE
        ${OpenCV_LIBS}
		
)

#-lstdc++fs
