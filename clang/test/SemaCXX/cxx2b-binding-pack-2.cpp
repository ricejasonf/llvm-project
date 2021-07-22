#include <cassert>

template <typename T>
struct type_ { };

template <typename ...T>
auto sum(T... t) { return (t + ...); }

struct my_struct {
	int a;
	int b;
	int c;
  int d;
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


template <typename T>
void decompose_tuple() {
  auto tup = T{{1, 2, 3, 6}};
  auto&& [x, ...rest, y] = tup;
  assert(x == 1);
  assert((rest + ...) == 5);
  assert(y == 6);

  ((void)type_<int>(type_<decltype(rest)>{}), ...);

  T arrtup[2] = {T{{1, 2, 3, 6}},
                 T{{7, 9, 10, 11}}};
  int sum = 0;
  for (auto [...xs] : arrtup) {
    sum += (xs + ...);
  }
  assert(sum == (1 + 2 + 3 + 6) +
                (7 + 9 + 10 + 11));
}

template <typename T>
void decompose_struct() {
  T obj{1, 2, 3, 6};
  auto [x, ...rest, y] = obj;
  assert(x == 1);
  assert((rest + ...) == 5);
  assert(y == 6);
}

template <typename T>
void decompose_array() {
  // previously unable to use non-dependent array here
  // Fixes https://bugs.llvm.org/show_bug.cgi?id=45964
  int arr[4] = {1, 2, 3, 6};
  auto [x, ...rest, y] = arr;
  assert(x == 1);
  assert((rest + ...) == 5);
  assert(y == 6);

  static_assert(sizeof...(rest) == 2);
  int size = sizeof...(rest);
  assert(size == 2);
  T arr2[sizeof...(rest)] = {rest...};
  auto [...pack] = arr2;
  assert(sum(pack...) == 5);
}

int main() {
  decompose_array<int>();
  decompose_tuple<fake_tuple>();
  decompose_struct<my_struct>();
}
