add_executable(porch_tensor_test tensor_test.cpp)
target_link_libraries(porch_tensor_test PRIVATE porch::porch)

add_test(NAME porch.tensor COMMAND porch_tensor_test)
