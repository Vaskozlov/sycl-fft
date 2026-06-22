#pragma once

#include <stdexcept>
#include <string>
#include <syclfft/common.hpp>

namespace test_utils
{
    inline void check(bool condition, const std::string &message)
    {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    inline syclfft::plan_options make_plan_options(
        syclfft::placement placement = syclfft::placement::out_of_place,
        syclfft::normalization normalization = syclfft::normalization::none,
        syclfft::provider provider = syclfft::provider::automatic,
        syclfft::planning_mode planning = syclfft::planning_mode::estimate)
    {
        syclfft::plan_options options;
        options.placement = placement;
        options.normalization = normalization;
        options.preferred_provider = provider;
        options.planning = planning;
        return options;
    }

    template <class Function>
    inline void expect_error(syclfft::error_code code, Function &&function)
    {
        try {
            function();
        } catch (const syclfft::exception &ex) {
            check(ex.code() == code, "wrong exception code");
            return;
        }
        throw std::runtime_error("expected syclfft::exception");
    }
} // namespace test_utils
