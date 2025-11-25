cmake_minimum_required(VERSION 3.15.0)

set(project_name "AiFrameworkSim")
project(${project_name})

# compile_commands.json 생성
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Conan 2 (CMakeDeps) 가 만들어준 패키지 설정들 위치
#   build/Release/generators 안에 boost / spdlog / nlohmann_json / robin_hood cmake 파일들 있음
#list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/Release/generators")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/Debug/generators")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/Release/generators")
endif()

# ===== Conan 패키지 =====
# conanfile.txt:
# [requires]
# boost/1.83.0
# robin-hood-hashing/3.11.5
# spdlog/1.11.0 (예시)
# nlohmann_json/3.11.2
#
# 위 기준의 CMake target 이름들
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(spdlog          CONFIG REQUIRED)
find_package(nlohmann_json   CONFIG REQUIRED)
find_package(robin_hood      CONFIG REQUIRED)

if (TARGET robin_hood::robin_hood)
    get_target_property(ROBIN_HOOD_INCLUDE_DIRS robin_hood::robin_hood INTERFACE_INCLUDE_DIRECTORIES)
    if (ROBIN_HOOD_INCLUDE_DIRS)
        include_directories(${ROBIN_HOOD_INCLUDE_DIRS})
    endif()
endif()

# ===== 옵션 / 모듈 경로 =====
option(USE_RAMULATOR "USE_RAMULATOR" ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/extern")

add_definitions(-DFMT_HEADER_ONLY=1)

# ===== C++ 설정 =====
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNX_ML 1)
set(JSON_BuildTests OFF CACHE INTERNAL "")

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
set(LIBRARY_OUTPUT_PATH    "${CMAKE_SOURCE_DIR}/build/lib")


message(STATUS "BINARY DIR: ${CMAKE_BINARY_DIR}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

# ===== 소스 / 외부 라이브러리들 =====
# 메인 소스
add_subdirectory("${PROJECT_SOURCE_DIR}/src")

# ramulator
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/ramulator_custom")

# ramulator2
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/ramulator2")
include_directories("${PROJECT_SOURCE_DIR}/extern/ramulator2/src")
include_directories("${PROJECT_SOURCE_DIR}/extern/ramulator2/resources/ndp_wrappers")

# booksim
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/booksim")

# protobuf
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/protobuf/cmake" EXCLUDE_FROM_ALL)
set_target_properties(libprotobuf PROPERTIES FOLDER "external/protobuf")
set_target_properties(protoc      PROPERTIES FOLDER "external/protobuf")

# onnx
add_definitions(-DONNX_NAMESPACE=onnx)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/onnx" EXCLUDE_FROM_ALL)
set_target_properties(onnx PROPERTIES FOLDER "external/onnx")

# ===== simulator / simulator_lib 타깃에 include + link =====
# (simulator, simulator_lib 타깃은 src/CMakeLists.txt 안에서 생성된다고 가정)

# 실행 파일
target_include_directories(Simulator PUBLIC ${ONNX_INCLUDE_DIRS})
target_include_directories(Simulator PRIVATE ${PROJECT_SOURCE_DIR}/extern/ramulator2/src)
target_include_directories(Simulator PRIVATE ${PROJECT_SOURCE_DIR}/extern/ramulator2/resources/ndp_wrappers)
target_include_directories(Simulator PRIVATE ${PROJECT_SOURCE_DIR}/extern/booksim/include/booksim2)
target_link_libraries(Simulator
    PRIVATE
        ramulator1
        booksim2
        ramulator
        ${PROTOBUF_LIB}
        onnx_proto
        Boost::program_options
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        stdc++fs           # 필요 없으면 이 줄은 나중에 지워도 됨
)

# 라이브러리
target_include_directories(Simulator_lib PUBLIC ${ONNX_INCLUDE_DIRS})
target_include_directories(Simulator_lib PRIVATE ${PROJECT_SOURCE_DIR}/extern/ramulator2/src)
target_include_directories(Simulator_lib PRIVATE ${PROJECT_SOURCE_DIR}/extern/ramulator2/resources/ndp_wrappers)
target_include_directories(Simulator_lib PRIVATE ${PROJECT_SOURCE_DIR}/extern/booksim/include/booksim2)
target_link_libraries(Simulator_lib
    PRIVATE
        ramulator1
        booksim2
        ramulator
        ${PROTOBUF_LIB}
        onnx_proto
        Boost::program_options
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        stdc++fs
)

# ===== 테스트 =====
enable_testing()
add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
