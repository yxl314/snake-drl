set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_snake_core snake_bindings.cpp)
target_link_libraries(_snake_core PRIVATE snake_core)
snake_configure_warnings(_snake_core)

install(TARGETS _snake_core LIBRARY DESTINATION snake_drl)
