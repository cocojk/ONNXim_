cmake_minimum_required(VERSION 3.15)

# 실행파일 이름
set(BINARY Simulator_test)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

# ---------- GoogleTest 외부 프로젝트 ----------
include(ExternalProject)
include(GoogleTest)

ExternalProject_Add(GoogleTest
    URL https://github.com/google/googletest/archive/release-1.8.1.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/lib
    # 굳이 ABI 꼬이게 하는 옵션 넣지 말고 CMake 기본 설정 사용
    CMAKE_ARGS -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    INSTALL_COMMAND ""
)

# ---------- spdlog (Conan에서 온 거 사용) ----------
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json   CONFIG REQUIRED)
find_package(robin_hood      CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# ---------- 테스트 소스 ----------
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false
    "*.h"
    "*.cc"
)

add_executable(${BINARY} ${TEST_SOURCES})

# 시뮬레이터 쪽 코드 / ONNX 헤더 include
target_include_directories(${BINARY}
    PUBLIC
        ${ONNX_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src
)

# spdlog 헤더 경로를 테스트 타깃에 명시적으로 추가
get_target_property(SPDLOG_INCLUDE_DIRS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)
if (SPDLOG_INCLUDE_DIRS)
    target_include_directories(${BINARY} PUBLIC ${SPDLOG_INCLUDE_DIRS})
endif()


# ---------- 링크 ----------
target_link_libraries(${BINARY}
    PRIVATE
        ramulator1
        booksim2
        ramulator
        Simulator_lib       # 시뮬레이터 라이브러리
        spdlog::spdlog      # Conan spdlog
        nlohmann_json::nlohmann_json
        onnx_proto
        Boost::program_options
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        pthread
        stdc++fs
)


