cmake_minimum_required(VERSION 3.20)

project(DNSS LANGUAGES CXX)

option(DNSS_ENABLE_CUDA "Enable CUDA acceleration for DNSS rotational feature computation" OFF)

add_library(dnss NSS.cpp NSS.h)
target_include_directories(dnss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(dnss PUBLIC cxx_std_17)

if(DNSS_ENABLE_CUDA)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
	if(NOT CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES native)
	endif()
	target_sources(dnss PRIVATE dnss_cuda.cu)
	target_compile_definitions(dnss PRIVATE DNSS_HAS_CUDA=1)
	target_link_libraries(dnss PUBLIC CUDA::cudart)
	set_target_properties(dnss PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
