// Copyright 2019 Eric Izoita (nytopop)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to
// do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
use std::mem;

pub trait Set<T> {
    /// Add val to the set.
    ///
    /// If the set did not have this value present, `true` is returned.
    ///
    /// If the set did have this value present, `false` is returned.
    fn set_insert(&mut self, t: T) -> bool;
}

impl<T: Ord> Set<T> for Vec<T> {
    /// O(n + log n)
    fn set_insert(&mut self, val: T) -> bool {
        self.binary_search(&val)
            .map_err(|i| self.insert(i, val))
            .is_err()
    }
}

pub trait Map<K, V> {
    /// Insert a value at key.
    ///
    /// Returns `None` if the key doesn't exist, and `Some(old_value)` when
    /// it does.
    fn map_insert(&mut self, key: K, val: V) -> Option<V>;

    /// Remove the value at key.
    ///
    /// Returns `None` if the key doesn't exist.
    fn map_remove(&mut self, key: &K) -> Option<(K, V)>;

    /// Lookup the value at key.
    ///
    /// Returns `None` if the key doesn't exist.
    fn map_lookup(&self, key: &K) -> Option<&V>;

    /// Find the smallest key that is greater than or equal to key, wrapping
    /// to zero if there isn't one.
    ///
    /// Returns `None` if the map is empty.
    fn find_gte(&self, key: &K) -> Option<&V>;
}

#[inline]
pub fn first<L, R>(tup: &(L, R)) -> &L {
    &tup.0
}

#[inline]
pub fn second<L, R>(tup: &(L, R)) -> &R {
    &tup.1
}

impl<K: Ord, V> Map<K, V> for Vec<(K, V)> {
    /// O(n + log n)
    fn map_insert(&mut self, key: K, val: V) -> Option<V> {
        match self.binary_search_by_key(&&key, first) {
            Err(i) => {
                self.insert(i, (key, val));
                Err(())
            }

            Ok(i) => Ok(mem::replace(
                &mut unsafe { self.get_unchecked_mut(i) }.1,
                val,
            )),
        }
        .ok()
    }

    /// O(n + log n)
    fn map_remove(&mut self, key: &K) -> Option<(K, V)> {
        self.binary_search_by_key(&key, first)
            .map(|i| self.remove(i))
            .ok()
    }

    /// O(log n)
    fn map_lookup(&self, key: &K) -> Option<&V> {
        self.binary_search_by_key(&key, first)
            .map(|i| unsafe { self.get_unchecked(i) })
            .map(second)
            .ok()
    }

    /// O(log n)
    fn find_gte(&self, key: &K) -> Option<&V> {
        let checked = |i| match self.len() {
            n if n == 0 => None,
            n if n == i => Some(0),
            _ => Some(i),
        };

        self.binary_search_by_key(&key, first)
            .map_or_else(checked, Some)
            .map(|i| unsafe { self.get_unchecked(i) })
            .map(second)
    }
}

/// An iterator that can have additional elements prepended to it.
pub struct Prependable<I, T> {
    inner: I,
    pre: Vec<T>,
}

impl<I: Iterator<Item = T>, T> Iterator for Prependable<I, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.pre.pop().or_else(|| self.inner.next())
    }
}

impl<I: Iterator<Item = T>, T> Prependable<I, T> {
    pub fn new(inner: I) -> Self {
        Self { inner, pre: vec![] }
    }

    pub fn push_front(&mut self, elt: T) {
        self.pre.push(elt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_set_insert() {
        let mut set = Vec::new();

        assert!(set.set_insert(0));
        assert!(!set.set_insert(0));
        assert!(set.set_insert(5));
        assert!(!set.set_insert(5));
        assert!(set.set_insert(3));
        assert!(!set.set_insert(3));
        assert_eq!(vec![0, 3, 5], set);
    }

    #[test]
    fn vec_map_insert() {
        let mut map = (0..2).map(|x| (x, x)).collect::<Vec<_>>();

        assert_eq!(None, map.map_insert(2, 42));
        assert_eq!(vec![(0, 0), (1, 1), (2, 42)], map);
        assert_eq!(Some(42), map.map_insert(2, 24));
        assert_eq!(vec![(0, 0), (1, 1), (2, 24)], map);
    }

    #[test]
    fn vec_map_remove() {
        let mut map = (0..2).map(|x| (x, x)).collect::<Vec<_>>();

        assert_eq!(None, map.map_remove(&4));
        assert_eq!(Some((1, 1)), map.map_remove(&1));
        assert_eq!(vec![(0, 0)], map);
    }

    #[test]
    fn vec_map_lookup() {
        let map = (0..2).map(|x| (x, x)).collect::<Vec<_>>();

        assert_eq!(Some(&1), map.map_lookup(&1));
        assert_eq!(None, map.map_lookup(&3));
    }

    #[test]
    fn vec_map_find_gte() {
        let mut map = Vec::default();

        assert_eq!(None, map.find_gte(&0));
        assert_eq!(None, map.map_insert(1, 2));
        assert_eq!(None, map.map_insert(2, 3));
        assert_eq!(None, map.map_insert(3, 4));
        assert_eq!(vec![(1, 2), (2, 3), (3, 4)], map);

        assert_eq!(Some(&2), map.find_gte(&0));
        assert_eq!(Some(&2), map.find_gte(&1));
        assert_eq!(Some(&3), map.find_gte(&2));
        assert_eq!(Some(&4), map.find_gte(&3));
        assert_eq!(Some(&2), map.find_gte(&4));
    }
}
