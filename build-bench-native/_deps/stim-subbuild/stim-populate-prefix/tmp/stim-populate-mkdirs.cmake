# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-src")
  file(MAKE_DIRECTORY "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-src")
endif()
file(MAKE_DIRECTORY
  "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-build"
  "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-subbuild/stim-populate-prefix"
  "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-subbuild/stim-populate-prefix/tmp"
  "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-subbuild/stim-populate-prefix/src/stim-populate-stamp"
  "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-subbuild/stim-populate-prefix/src"
  "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-subbuild/stim-populate-prefix/src/stim-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-subbuild/stim-populate-prefix/src/stim-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/aryakakodkar/Documents/qerasure/build-bench-native/_deps/stim-subbuild/stim-populate-prefix/src/stim-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
