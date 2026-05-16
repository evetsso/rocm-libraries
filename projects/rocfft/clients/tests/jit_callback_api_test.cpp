// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Argument-validation tests for the JIT-callback registration API.

#include "../../shared/params_gen.h"
#include "../../shared/test_params.h"
#include "rocfft/rocfft.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

// Placeholder bitcode; entry points only inspect pointer/length.
static unsigned char kDummyBitcode[16] = {};

struct ScopedDescription
{
    ScopedDescription()
    {
        EXPECT_EQ(rocfft_plan_description_create(&desc), rocfft_status_success);
    }
    ScopedDescription(const ScopedDescription&) = delete;
    ScopedDescription& operator=(const ScopedDescription&) = delete;
    ~ScopedDescription()
    {
        if(desc)
            rocfft_plan_description_destroy(desc);
    }
    rocfft_plan_description desc = nullptr;
};

// Load and store share validation logic; drive both through one helper.
// Explicit extern "C" typedef keeps function-pointer linkage unambiguous.
extern "C" {
using SetCallbackFn = rocfft_status (*)(rocfft_plan_description description,
                                        const char*             symbol_name,
                                        void*                   bitcode_data,
                                        size_t                  bitcode_len_bytes,
                                        void**                  cb_data,
                                        size_t                  shared_mem_bytes);
}

static void check_argument_validation(SetCallbackFn set_cb)
{
    ScopedDescription d;
    ASSERT_NE(d.desc, nullptr);

    void* const  bitcode = kDummyBitcode;
    const size_t len     = sizeof(kDummyBitcode);

    // null description
    EXPECT_EQ(set_cb(nullptr, "valid_name", bitcode, len, nullptr, 0),
              rocfft_status_invalid_arg_value);

    // shared_mem_bytes != 0 is reserved
    EXPECT_EQ(set_cb(d.desc, "valid_name", bitcode, len, nullptr, /*shared_mem_bytes=*/16),
              rocfft_status_invalid_arg_value);

    // symbol name must be [A-Za-z_][A-Za-z0-9_]*
    for(const char* bad : {"0starts_with_digit", "has space", "has-dash", "has.dot", "name$dollar"})
    {
        EXPECT_EQ(set_cb(d.desc, bad, bitcode, len, nullptr, 0), rocfft_status_invalid_arg_value)
            << "symbol: \"" << bad << "\"";
    }
    for(const char* ok : {"load_cb", "_underscore", "Mixed_Case_42", "x"})
    {
        EXPECT_EQ(set_cb(d.desc, ok, bitcode, len, nullptr, 0), rocfft_status_success)
            << "symbol: \"" << ok << "\"";
    }

    // null symbol, null bitcode, or zero length clears the registration
    EXPECT_EQ(set_cb(d.desc, nullptr, bitcode, len, nullptr, 0), rocfft_status_success);
    EXPECT_EQ(set_cb(d.desc, "valid_name", nullptr, len, nullptr, 0), rocfft_status_success);
    EXPECT_EQ(set_cb(d.desc, "valid_name", bitcode, 0, nullptr, 0), rocfft_status_success);

    // re-register after clearing
    EXPECT_EQ(set_cb(d.desc, "fresh_after_clear", bitcode, len, nullptr, 0), rocfft_status_success);
}

TEST(rocfft_JitCallbackApi, set_load_callback_argument_validation)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    check_argument_validation(rocfft_plan_description_set_load_callback);
}

TEST(rocfft_JitCallbackApi, set_store_callback_argument_validation)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    check_argument_validation(rocfft_plan_description_set_store_callback);
}

// Registrations on distinct descriptions must not alias
TEST(rocfft_JitCallbackApi, registrations_are_independent_per_description)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    ScopedDescription a;
    ScopedDescription b;
    ASSERT_NE(a.desc, nullptr);
    ASSERT_NE(b.desc, nullptr);
    ASSERT_NE(a.desc, b.desc);

    void* const  bitcode = kDummyBitcode;
    const size_t len     = sizeof(kDummyBitcode);

    EXPECT_EQ(rocfft_plan_description_set_load_callback(a.desc, "load_a", bitcode, len, nullptr, 0),
              rocfft_status_success);
    EXPECT_EQ(rocfft_plan_description_set_load_callback(b.desc, "load_b", bitcode, len, nullptr, 0),
              rocfft_status_success);

    // overwrite on a, store on b
    EXPECT_EQ(
        rocfft_plan_description_set_load_callback(a.desc, "load_a2", bitcode, len, nullptr, 0),
        rocfft_status_success);
    EXPECT_EQ(
        rocfft_plan_description_set_store_callback(b.desc, "store_b", bitcode, len, nullptr, 0),
        rocfft_status_success);

    // clear b, then mutate a
    EXPECT_EQ(rocfft_plan_description_set_load_callback(b.desc, nullptr, nullptr, 0, nullptr, 0),
              rocfft_status_success);
    EXPECT_EQ(
        rocfft_plan_description_set_store_callback(a.desc, "store_a", bitcode, len, nullptr, 0),
        rocfft_status_success);
}

// Load and store coexist; overwrite is last-writer-wins; clear is idempotent
TEST(rocfft_JitCallbackApi, load_and_store_coexist_and_overwrite)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    ScopedDescription d;
    ASSERT_NE(d.desc, nullptr);

    void* const  bitcode = kDummyBitcode;
    const size_t len     = sizeof(kDummyBitcode);

    // coexist
    EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, "load_a", bitcode, len, nullptr, 0),
              rocfft_status_success);
    EXPECT_EQ(
        rocfft_plan_description_set_store_callback(d.desc, "store_a", bitcode, len, nullptr, 0),
        rocfft_status_success);

    // overwrite each
    EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, "load_b", bitcode, len, nullptr, 0),
              rocfft_status_success);
    EXPECT_EQ(
        rocfft_plan_description_set_store_callback(d.desc, "store_b", bitcode, len, nullptr, 0),
        rocfft_status_success);

    // clearing one field does not disturb the other
    EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, nullptr, nullptr, 0, nullptr, 0),
              rocfft_status_success);
    EXPECT_EQ(
        rocfft_plan_description_set_store_callback(d.desc, "store_c", bitcode, len, nullptr, 0),
        rocfft_status_success);
    EXPECT_EQ(rocfft_plan_description_set_store_callback(d.desc, nullptr, nullptr, 0, nullptr, 0),
              rocfft_status_success);
    EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, "load_c", bitcode, len, nullptr, 0),
              rocfft_status_success);

    // idempotent clear
    EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, nullptr, nullptr, 0, nullptr, 0),
              rocfft_status_success);
    EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, nullptr, nullptr, 0, nullptr, 0),
              rocfft_status_success);
}

// Symbol-name regex edge cases beyond the basic validation test
TEST(rocfft_JitCallbackApi, symbol_name_length_and_edge_cases)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    ScopedDescription d;
    ASSERT_NE(d.desc, nullptr);

    void* const  bitcode = kDummyBitcode;
    const size_t len     = sizeof(kDummyBitcode);

    // single-character valid names
    for(const char* ok : {"x", "_"})
    {
        EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, ok, bitcode, len, nullptr, 0),
                  rocfft_status_success)
            << "symbol: \"" << ok << "\"";
    }

    // digits valid in non-leading positions
    for(const char* ok : {"cb42", "_42", "a1b2c3", "X9_y0"})
    {
        EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, ok, bitcode, len, nullptr, 0),
                  rocfft_status_success)
            << "symbol: \"" << ok << "\"";
    }

    // leading digit always rejected as invalid_arg_value
    for(const char* bad : {"42", "1cb", "9_underscore"})
    {
        EXPECT_EQ(rocfft_plan_description_set_load_callback(d.desc, bad, bitcode, len, nullptr, 0),
                  rocfft_status_invalid_arg_value)
            << "symbol: \"" << bad << "\"";
    }

    // long valid name regression guard
    {
        std::string long_name(256, '_');
        long_name[0] = 'a';
        EXPECT_EQ(rocfft_plan_description_set_load_callback(
                      d.desc, long_name.c_str(), bitcode, len, nullptr, 0),
                  rocfft_status_success);

        // clear before long_name goes out of scope; desc stores
        // symbol_name as a non-owning pointer
        EXPECT_EQ(
            rocfft_plan_description_set_load_callback(d.desc, nullptr, nullptr, 0, nullptr, 0),
            rocfft_status_success);
    }
}
