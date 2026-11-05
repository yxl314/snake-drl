include_guard(GLOBAL)

include(CheckCXXCompilerFlag)

# Applies the project warning policy to one target. Reads the SNAKE_* options
# from the enclosing scope (they are cache variables, so a sibling directory
# such as bindings/ sees the same values).
function(snake_configure_warnings target)
  if(MSVC)
    set(flags /W4 /permissive-)
    if(SNAKE_WARNINGS_AS_ERRORS)
      list(APPEND flags /WX)
    endif()
    target_compile_options(${target} PRIVATE ${flags})
    return()
  endif()

  set(flags -Wall -Wextra -Wpedantic -Wshadow)
  if(SNAKE_STRICT_WARNINGS)
    foreach(flag IN ITEMS
        -Wconversion
        -Wsign-conversion
        -Wold-style-cast
        -Wuseless-cast
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
        -Wcast-qual
        -Wzero-as-null-pointer-constant
        -Wnon-virtual-dtor)
      string(MAKE_C_IDENTIFIER "SNAKE_FLAG${flag}" flag_var)
      check_cxx_compiler_flag(${flag} ${flag_var})
      if(${flag_var})
        list(APPEND flags ${flag})
      endif()
    endforeach()
  endif()
  if(SNAKE_WARNINGS_AS_ERRORS)
    list(APPEND flags -Werror)
  endif()
  target_compile_options(${target} PRIVATE ${flags})
endfunction()
