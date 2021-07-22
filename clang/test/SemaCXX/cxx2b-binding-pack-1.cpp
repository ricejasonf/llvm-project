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

int main() {
  decompose_array();
  decompose_struct();
  decompose_tuple();
}

