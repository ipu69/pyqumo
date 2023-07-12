#include <gtest/gtest.h>
#include <cqumo/functions.h>


TEST(TestFunctions, ContextFunctor) {
    auto fn = [](void* reg) -> double {
        return (int)(*((double*)reg));
    };
    double val{4.2};
    cqumo::ContextFunctor functor(fn, &val);
    ASSERT_EQ(4, functor());
    val = 100.56;
    ASSERT_EQ(100, functor());
}


TEST(TestFunctions, makeDblFn) {
    double val{42};
    auto ctx_fn = [](void *reg) -> double {
        return *(double*)reg;
    };
    auto fn = cqumo::makeDblFn(ctx_fn, &val);
    ASSERT_EQ(fn(), 42);
    val = 34;
    ASSERT_EQ(fn(), 34);
}