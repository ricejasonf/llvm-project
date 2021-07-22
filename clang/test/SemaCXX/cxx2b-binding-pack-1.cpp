//#define assert(...) ((__VA_ARGS__) ? ((void)0) : throw 42)
#include <cassert>

template <typename T>
struct type_ { };

template <typename ...T>
auto sum(T... t) { return (t + ...); }

struct my_struct {
	int a;
	int b;
	int c;
  int &d;
};

struct fake_tuple {
  int arr[4] = {1, 2, 3, 6};

  template <unsigned i>
  int get() {
    return arr[i];
  }
};

namespace std {
  template <typename T>
  struct tuple_size;
  template <unsigned i, typename T>
  struct tuple_element;

  template <>
  struct tuple_size<fake_tuple> {
    static constexpr unsigned value = 4;
  };

  template <unsigned i>
  struct tuple_element<i, fake_tuple> {
    using type = int;
  };
}

void decompose_tuple() {
  auto tup = fake_tuple{{1, 2, 3, 6}};
  auto&& [x, ...rest, y] = tup;
  assert(x == 1);
  assert((rest + ...) == 5);
  assert(y == 6);

  ((void)type_<int>(type_<decltype(rest)>{}), ...);
}

void decompose_struct() {
  int d = 6;
  my_struct obj{1, 2, 3, d};
  auto&& [x, ...rest] = obj;
  assert(x == 1);
  assert((rest + ...) == 11);
}

void decompose_array() {
  int arr[4] = {1, 2, 3, 6};
  auto [x, ...rest, y] = arr;
  assert(x == 1);
  assert((rest + ...) == 5);
  assert(y == 6);

  static_assert(sizeof...(rest) == 2);
  int size = sizeof...(rest);
  assert(size == 2);
  int arr2[sizeof...(rest)] = {rest...}; // crash
  auto [...pack] = arr2;
  assert(sum(pack...) == 5);
}

int nine_implicit_template_instantiations() {
    int arr[4] = {1, 2, 3, 6};

    // Switch
    switch (auto [...rest, last] = arr; (rest + ...)) {
      case 6: if ((rest + ...) == last) break;
      // else fall through
      default: return 1;
    }

    // If
    if (auto [...rest, last] = arr; (rest + ...) == 6)
      if ((rest + ...) != last)
        return 1;

    // If-else body (not compound statement).
    if (true)
      auto [...rest, last] = arr;
    else
      auto [...rest, last] = arr;

    // For
    for (auto [...rest, last] = arr; true; true) {
      if ((rest + ...) != last) return 1;
      break;
    }

    // For body (not compound statement).
    for ((void)(0); false; (void)(0))
      auto [...rest, last] = arr;

    // For-Ranged
    for (auto [...rest, last] : (int[1][4]){{1, 2, 3, 6}})
      break;

    // For-Ranged body (not compound statement)
    for (int x : arr)
      auto [...rest] = arr;

    // Invariably, any implicit template will be cleaned
    // up by the end the function body (compound statement).
    // This last test adds one additional instantiation so
    // we can count them properly.
    auto [...rest, last] = arr;
    if ((rest + ...) != last) return 1;
    return 0;
}


struct C { char j; int l; };

int g() {
    auto [ ... i ] = C{ 'x', 42 }; // #1
    return ( [c = i] () {
        // The local class L is a templated entity because
        // it is defined within the implicit template region
        // created at #1.
        struct L {
            int f() requires (sizeof(c) == 1) { return 1; }
            int f() requires (sizeof(c) != 1) { return 2; }
        };
        return L{}.f();
    } () + ...  + 0 );
}

int v = g(); // OK, v == 3

int main() {
  decompose_array();
  decompose_struct();
  decompose_tuple();
}

