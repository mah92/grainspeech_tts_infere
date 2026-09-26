cmake_minimum_required(VERSION 3.10)
project(GrainspeechTTSInfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

find_package(ICU REQUIRED COMPONENTS uc i18n io)

# Find ONNX Runtime
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)
find_path(ONNX_RUNTIME_INCLUDE onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime)

if(NOT ONNX_RUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()
if(NOT ONNX_RUNTIME_INCLUDE)
    message(FATAL_ERROR "ONNX Runtime headers not found")
endif()

message(STATUS "ONNX Runtime lib: ${ONNX_RUNTIME_LIB}")
message(STATUS "ONNX Runtime include: ${ONNX_RUNTIME_INCLUDE}")

# Find the espeak-ng library
find_package(PkgConfig REQUIRED)
pkg_check_modules(ESPEAK REQUIRED espeak-ng)

# ---- Dependency paths (NormalizeText at root, deps inside) ----
set(NORMALIZE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NormalizeText)
set(HAZM_CPP_DIR ${NORMALIZE_DIR}/hazm_cpp)
set(EZAFE_DIR ${NORMALIZE_DIR}/persian-ezafe-albert-cpp)

# ---- normalize sources ----
set(NORMALIZE_SOURCES
    ${NORMALIZE_DIR}/grain_phones.cpp
    ${NORMALIZE_DIR}/normalize.cpp
    ${NORMALIZE_DIR}/replacements/replacements.cpp
    ${NORMALIZE_DIR}/replacements/replacementHelpers.cpp
    ${NORMALIZE_DIR}/language_detector/language_detector.cpp
    ${NORMALIZE_DIR}/espeak_phonemize/phonemize.cpp
    ${NORMALIZE_DIR}/espeak_phonemize/phoneme_ids.cpp
    ${NORMALIZE_DIR}/shakkelha/shakkelha.cpp
    ${NORMALIZE_DIR}/nawar-halabi/ar_phonemizer.cpp
)

# ---- hazm_cpp sources ----
set(HAZM_SOURCES
    ${HAZM_CPP_DIR}/src/utils.cpp
    ${HAZM_CPP_DIR}/src/stemmer.cpp
    ${HAZM_CPP_DIR}/src/conjugation.cpp
    ${HAZM_CPP_DIR}/src/word_tokenizer.cpp
    ${HAZM_CPP_DIR}/src/lemmatizer.cpp
    ${HAZM_CPP_DIR}/src/stopwords.cpp
)

# ---- ezafe_detector sources ----
set(EZAFE_SOURCES
    ${EZAFE_DIR}/src/ezafe_detector.cpp
    ${EZAFE_DIR}/third_party/sentencepiece_minimal/sentencepiece_minimal.cpp
)

# ---- Homograph + Persian post-process sources ----
set(PERSIAN_POSTPROCESS_SOURCES
    ${NORMALIZE_DIR}/homograph/homograph_dict.cpp
    ${NORMALIZE_DIR}/homograph/homograph_disambiguator.cpp
    ${NORMALIZE_DIR}/persian_phoneme/persian_postprocess.cpp
)

# ---- Main executable ----
add_executable(${PROJECT_NAME}
    grainspeech_infer.cpp
    ${NORMALIZE_SOURCES}
    ${HAZM_SOURCES}
    ${EZAFE_SOURCES}
    ${PERSIAN_POSTPROCESS_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${NORMALIZE_DIR}
    ${NORMALIZE_DIR}/language_detector
    ${NORMALIZE_DIR}/json_cpp
    ${NORMALIZE_DIR}/homograph
    ${NORMALIZE_DIR}/persian_phoneme
    ${HAZM_CPP_DIR}/include
    ${EZAFE_DIR}/src
    ${EZAFE_DIR}/third_party/sentencepiece_minimal
    ${ONNX_RUNTIME_INCLUDE}
)

# Define data directory for hazm_cpp (relative to cwd at runtime)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    HAZM_DATA_DIR="./assets/"
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE ICU::uc ICU::i18n ICU::io)
target_include_directories(${PROJECT_NAME} PRIVATE ${ESPEAK_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ESPEAK_LIBRARIES})
