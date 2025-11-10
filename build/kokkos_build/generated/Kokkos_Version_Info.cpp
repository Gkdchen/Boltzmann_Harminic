// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "Kokkos_Version_Info.hpp"

namespace Kokkos {
namespace Impl {

std::string GIT_BRANCH       = R"branch(develop)branch";
std::string GIT_COMMIT_HASH  = "a283d9332";
std::string GIT_CLEAN_STATUS = "CLEAN";
std::string GIT_COMMIT_DESCRIPTION =
    R"message(Merge pull request #8661 from tretre91/simd_sve_missing_cmake_include)message";
std::string GIT_COMMIT_DATE = "2025-11-09T20:31:30-10:00";

}  // namespace Impl

}  // namespace Kokkos
