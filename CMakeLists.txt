# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(l2fwdnv LANGUAGES CXX C CUDA)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENABLE_TESTING()



# FIXME: enable build for a generic target
ADD_COMPILE_DEFINITIONS("ALLOW_EXPERIMENTAL_API")

# Set the C, C++, and Fortran compilers
set(CMAKE_C_COMPILER "/opt/intel/oneapi/compiler/2023.2.1/linux/bin/icx")
set(CMAKE_CXX_COMPILER "/opt/intel/oneapi/compiler/2023.2.1/linux/bin/icpx")

# Set the compiler flags for C, C++, and Fortran
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -CC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

cmake_minimum_required(VERSION 3.0)
project(YourProject)

# Set the path to the SYCL include directory
set(SYCL_INCLUDE_DIR "/opt/intel/oneapi/compiler/2023.2.1/linux/include/")

# Include the SYCL directory
include_directories(${SYCL_INCLUDE_DIR})



# Set the path to the SPIR-V include directory
set(SPIRV_INCLUDE_DIR "/opt/intel/oneapi/compiler/2023.2.1/linux/include/CL/__spirv")

# Include the SPIR-V directory
include_directories(${SPIRV_INCLUDE_DIR})

# Include other necessary directories
include_directories("/opt/intel/oneapi/compiler/2023.2.1/linux/include/sycl")


# Set the path to the DPC++ include directory
set(DPCPP_INCLUDE_DIR "/opt/intel/oneapi/dpcpp-ct/latest/include")

# Include the DPC++ directory
include_directories(${DPCPP_INCLUDE_DIR})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

 
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-unnamed-lambda -sycl-std=2020")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl  -fsycl-unnamed-lambda -sycl-std=2020")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-unnamed-lambda")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin;/opt/intel/oneapi/compiler/2023.2.1/linux/lib;/usr/lib64/gcc/x86_64-suse-linux/13;/usr/lib64;/lib64;/usr/lib/x86_64-linux-gnu;/usr/x86_64-suse-linux/lib;/lib;/usr/lib;/opt/intel/oneapi/compiler/2023.2.0/linux/lib;/opt/intel/oneapi/compiler/2023.2.1/linux/lib")



include_directories("/home/user/l2fwd-intl/include")

########################################### OPTIONS ##################################################

OPTION(BUILD_DOCS "Generate Doxygen documentation" OFF)
OPTION(GTEST_POPULATE_CMAKE_TESTS "Populate CMake tests with GTest" ON)
OPTION(CUDA_ENABLED "Whether to enable CUDA for L2FWDNV" ON) #FIXME We probably always need CUDA
OPTION(IPV4_UDP_MODE "Encapsulate eCPRI packets in IPv4/UDP" OFF)
OPTION(PROFILE_LOGS "Periodically dump profiling and monitoring statistics" OFF)
option(ENABLE_NVTX  "Enable use of NVTX" OFF)

############################################ CUDA ###################################################

#FIND_PACKAGE(CUDA 12.2 REQUIRED)
FIND_LIBRARY(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
IF (NOT CUDA_DRIVER_LIBRARY)
    FIND_LIBRARY(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs compat)
ENDIF ()
MARK_AS_ADVANCED(CUDA_DRIVER_LIBRARY)



set(ONEAPI_LIBRARY "${ONEAPI_LIBRARY} -L/opt/intel/oneapi/compiler/2023.2.1/linux/lib")

#FIND_LIBRARY(ONEAPI_LIBRARY
#             NAMES intel64_lin intel64 lib
#             HINTS /opt/intel/oneapi/ 
#             PATH_SUFFIXES compiler/2023.2.1/linux/compile mkl/2023.2.0/lib dal/2023.2.0/lib compiler/2023.2.0/linux)

############################################ DPDK ###################################################

SET(DPDK_TARGET dpdk_target)
SET(DPDK_TARGET_ARCH x86_64-native-linuxapp-gcc)
# Build on ARM
# SET(DPDK_TARGET_ARCH aarch64-build-gcc)
SET(DPDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/dpdk)
message( ${DPDK_PATH})
SET(DPDK_TARGET_PATH ${DPDK_PATH}/${DPDK_TARGET_ARCH})
message( ${DPDK_TARGET_PATH} )
SET(DPDK_TARGET_INSTALL_PATH ${DPDK_TARGET_PATH}/install)
STRING(CONCAT DPDK_DISABLED_DRIVERS
        "baseband/*,"
        "bus/ifpga/*,"
        "common/cpt,"
        "common/dpaax,"
        "common/iavf,"
        "common/octeontx,"
        "common/octeontx2,"
        "crypto/nitrox,"
        "net/ark,"
        "net/atlantic,"
        "net/avp,"
        "net/axgbe,"
        "net/bnx2x,"
        "net/bnxt,"
        "net/cxgbe,"
        "net/e1000,"
        "net/ena,"
        "net/enic,"
        "net/fm10k,"
        "net/hinic,"
        "net/hns3,"
        "net/i40e,"
        "net/ixgbe,"
        "vdpa/ifc,"
        "net/igc,"
        "net/liquidio,"
        "net/netvsc,"
        "net/nfp,"
        "net/qede,"
        "net/sfc,"
        "net/thunderx,"
        "net/vdev_netvsc,"
        "net/vmxnet3,"
        "regex/octeontx2,"
	"net/mlx5,"
)

#SET(ENV{CFLAGS} "${CFLAGS} -I${CUDA_INCLUDE_DIRS} -I${CMAKE_CURRENT_SOURCE_DIR}/external/gdrcopy/include")
SET(ENV{CFLAGS} "${CFLAGS}")
EXECUTE_PROCESS(WORKING_DIRECTORY ${DPDK_PATH}
        COMMAND meson ${DPDK_TARGET_ARCH}
                --prefix=${DPDK_TARGET_INSTALL_PATH}
                -Dtests=false
                -Ddisable_drivers=${DPDK_DISABLED_DRIVERS}
                # -Dc_args="-I${CUDA_INCLUDE_DIRS} -I${CMAKE_CURRENT_SOURCE_DIR}/external/gdrcopy/include"
                --cross-file config/arm/arm64_armv8_linux_gcc
                -Dbuildtype=debug
		--reconfigure
        )

# Set RTE_PKTMBUF_HEADROOM to 0 for GPU memory best performance
EXECUTE_PROCESS(WORKING_DIRECTORY ${DPDK_PATH}
        COMMAND sed -ri "s,(RTE_PKTMBUF_HEADROOM ).*,\\10," config/rte_config.h
)

ADD_CUSTOM_TARGET(${DPDK_TARGET} ALL
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}
        COMMENT Building DPDK
        COMMAND ninja install
        VERBATIM)

SET(ENV{PKG_CONFIG_PATH} "${DPDK_TARGET_PATH}/meson-private")

EXECUTE_PROCESS(OUTPUT_VARIABLE DPDK_BUILD_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
        COMMAND pkg-config --cflags libdpdk.pc)

EXECUTE_PROCESS(OUTPUT_VARIABLE DPDK_LINK_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
        COMMAND pkg-config --static --libs libdpdk.pc)

SET(DPDK_LIBS "-Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl ${DPDK_LINK_FLAGS}")
SET(ONEAPI_LIBS "-lsycl")

############################################ GDRCopy ###################################################

#SET(GDRC_TARGET gdrcopy_lib)
#add_custom_target(
#   ${GDRC_TARGET}
#   COMMAND make
#   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/gdrcopy
#)
## gdrcopy makefile doesn't support make with multiple jobs
## dorce cmake to build DPDK first
#ADD_DEPENDENCIES(${GDRC_TARGET} ${DPDK_TARGET})

######################################### L2FWDNV ################################################

SET(TARGET l2fwdnv)

SET(L2FWDNV_SRCS
        src/kernel.dp.cpp
        src/main.cpp.dp.cpp
        src/utils.cpp
        src/pipeline.cpp.dp.cpp
)

ADD_EXECUTABLE(${TARGET} ${L2FWDNV_SRCS})
# Force cmake to build libgdrapi.so
#ADD_DEPENDENCIES(${TARGET} ${GDRC_TARGET})
#ADD_DEPENDENCIES(${TARGET})

# Link with nvToolsExt
if (ENABLE_NVTX)
    set(NVTX_LIB nvToolsExt)
    add_definitions(-DUSE_NVTX=1)
endif (ENABLE_NVTX)

TARGET_INCLUDE_DIRECTORIES(${TARGET} PUBLIC include ${CUDA_INCLUDE_DIRS})
#TARGET_LINK_LIBRARIES(l2fwdnv
#  PRIVATE
#    -Wl,--no-whole-archive
#    mlx5
#    ibverbs
#    pthread
#    numa
#    dl
#)
TARGET_LINK_LIBRARIES(${TARGET} ${DPDK_LIBS} ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} ${NVTX_LIB} -L/opt/intel/oneapi/compiler/2023.2.1/linux/lib/ ${ONEAPI_LIBS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O -std=c++17 ${CMAKE_COMMON_FLAGS} -g  -O0 ${DPDK_BUILD_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_COMMON_FLAGS} ${DPDK_BUILD_FLAGS}")

# Enable to debug CUDA kernels
#string(STRIP ${DPDK_BUILD_FLAGS} DPDK_STRIPPED_FLAGS)
#string(REGEX
#        REPLACE
#        "\-std\="
#        "\-Xcompiler \-std\="
#        DPDK_STRIPPED_FLAGS
#        ${DPDK_STRIPPED_FLAGS}
#)
#
#string(REGEX
#        REPLACE
#        "\-march\="
#        "\-Xcompiler \-march\="
#        DPDK_STRIPPED_FLAGS
#        ${DPDK_STRIPPED_FLAGS}
#)
#
#string(REGEX
#        REPLACE
#        "\-moutline\-atomics"
#        "\-Xcompiler \-moutline\-atomics"
#        DPDK_STRIPPED_FLAGS
#        ${DPDK_STRIPPED_FLAGS}
#)


SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_COMMON_FLAGS} ${DPDK_STRIPPED_FLAGS}") # -G -lineinfo")

message(STATUS "CMAKE_COMMON_FLAGS ${CMAKE_COMMON_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")
message(STATUS "DPDK_BUILD_FLAGS ${DPDK_BUILD_FLAGS}")

ADD_DEPENDENCIES(${TARGET} ${DPDK_TARGET})
INSTALL(TARGETS ${TARGET} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include/l2fwdnv)




# Find the DPDK libraries
#find_library(DPDK_EAL_LIB NAMES rte_eal PATHS "/usr/lib64" NO_DEFAULT_PATH)
#find_library(DPDK_MEMPOOL_LIB NAMES rte_mempool PATHS "/path/to/dpdk/libraries" NO_DEFAULT_PATH)
# Add other necessary DPDK libraries using the find_library command


# Link your executable to the DPDK libraries
#target_link_libraries( PRIVATE
#	    ${DPDK_EAL_LIB}
#	    #${DPDK_MEMPOOL_LIB}
#		 # Add other DPDK library variables here
#	    )
