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

#![feature(test, result_map_or_else)]

extern crate crc;

use std::{
    collections::HashSet,
    hash::{BuildHasher, BuildHasherDefault, Hash, Hasher},
};

type VecMap<K, V> = Vec<(K, V)>;

trait OrdMap<K, V> {
    fn ord_insert(&mut self, key: K, val: V) -> bool;
    fn ord_remove(&mut self, key: &K) -> Option<(K, V)>;
    fn find_gte(&self, key: &K) -> Option<&V>;
}

#[inline]
fn first<L, R>(tup: &(L, R)) -> &L {
    &tup.0
}

#[inline]
fn second<L, R>(tup: &(L, R)) -> &R {
    &tup.1
}

impl<K: Ord, V> OrdMap<K, V> for VecMap<K, V> {
    fn ord_insert(&mut self, key: K, val: V) -> bool {
        self.binary_search_by_key(&&key, first)
            .map_err(|i| self.insert(i, (key, val)))
            .is_err()
    }

    fn ord_remove(&mut self, key: &K) -> Option<(K, V)> {
        self.binary_search_by_key(&key, first)
            .map(|i| self.remove(i))
            .ok()
    }

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

#[derive(Default)]
pub struct DigestIEEE(u32);

impl Hasher for DigestIEEE {
    fn write(&mut self, bytes: &[u8]) {
        use crc::crc32::{update, IEEE_TABLE};

        self.0 = update(self.0, &IEEE_TABLE, bytes);
    }

    fn finish(&self) -> u64 {
        u64::from(self.0)
    }
}

type IEEE = BuildHasherDefault<DigestIEEE>;

/// A consistent hash ring with variable weight nodes.
#[derive(Clone)]
pub struct Ring<T: Hash + Eq + Clone, S = IEEE> {
    replicas: usize,      // default number of replicas for each node
    ring: VecMap<u64, T>, // the ring itself
    hash: S,              // selected build hasher
}

impl<T: Hash + Eq + Clone> Default for Ring<T> {
    fn default() -> Self {
        Self::new(10)
    }
}

impl<T: Hash + Eq + Clone> Ring<T> {
    /// Create a new ring.
    pub fn new(replicas: usize) -> Self {
        Ring {
            replicas,
            ring: VecMap::default(),
            hash: IEEE::default(),
        }
    }

    /// Create a new ring from an iterator of nodes.
    pub fn new_with_nodes<I>(replicas: usize, nodes: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        let mut ring = Ring::new(replicas);
        ring.insert_iter(nodes);
        ring
    }
}

impl<T: Hash + Eq + Clone, S: BuildHasher> Ring<T, S> {
    /// Create a new ring with the provided BuildHasher.
    pub fn new_with_hasher(replicas: usize, hash: S) -> Self {
        Ring {
            replicas,
            ring: VecMap::new(),
            hash,
        }
    }

    /// Create a new ring with the provided BuildHasher and iterator
    /// of nodes.
    pub fn new_with_hasher_and_nodes<I>(replicas: usize, hash: S, nodes: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        let mut ring = Ring::new_with_hasher(replicas, hash);
        ring.insert_iter(nodes);
        ring
    }

    /// Returns the number of nodes in the ring (not including any replicas).
    ///
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let ring = Ring::new_with_nodes(10, 0..32);
    /// assert_eq!(32, ring.len());
    /// ```
    pub fn len(&self) -> usize {
        self.ring.iter().map(second).collect::<HashSet<_>>().len()
    }

    /// Returns the number of replicas of the provided node, or 0 if it does
    /// not exist.
    ///
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new_with_nodes(12, 0..4);
    /// ring.insert_weight(&42, 9);
    /// assert_eq!(12, ring.weight(&3));
    /// assert_eq!(9, ring.weight(&42));
    /// ```
    pub fn weight(&self, node: &T) -> usize {
        self.ring.iter().filter(|(_, _node)| _node == node).count()
    }

    /// Insert nodes from an iterator.
    ///
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new(10);
    /// assert_eq!(0, ring.len());
    /// ring.insert_iter(0..14);
    /// assert_eq!(14, ring.len());
    /// ```
    pub fn insert_iter<I: Iterator<Item = T>>(&mut self, nodes: I) {
        nodes.for_each(|node| self.insert(&node))
    }

    /// Insert weighted nodes from an iterator.
    ///
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new(10);
    /// assert_eq!(0, ring.len());
    /// ring.insert_weight_iter((0..12).map(|x| (x, 4)));
    /// assert_eq!(12, ring.len());
    /// assert_eq!(4, ring.weight(&7));
    /// ```
    pub fn insert_weight_iter<I>(&mut self, nodes: I)
    where
        I: Iterator<Item = (T, usize)>,
    {
        nodes.for_each(|(node, replicas)| self.insert_weight(&node, replicas))
    }

    /// Insert a node into the ring with the default replica count.
    ///
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new(3);
    /// ring.insert(b"hello worldo");
    /// assert_eq!(1, ring.len());
    /// assert_eq!(3, ring.weight(b"hello worldo"));
    /// ```
    pub fn insert(&mut self, node: &T) {
        self.insert_weight(node, self.replicas)
    }

    /// Insert a node into the ring with the provided number of replicas.
    ///
    /// This can be used give some nodes more weight than others - nodes
    /// with more replicas will be selected for larger proportions of keys.
    ///
    /// If the provided node is already present in the ring with a lower
    /// replica count, the count is updated. If the existing count is higher,
    /// this method does nothing.
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new(3);
    /// ring.insert(b"hello worldo");
    /// ring.insert_weight(b"worldo hello", 9);
    /// assert_eq!(2, ring.len());
    /// assert_eq!(3, ring.weight(b"hello worldo"));
    /// assert_eq!(9, ring.weight(b"worldo hello"));
    /// ```
    pub fn insert_weight(&mut self, node: &T, replicas: usize) {
        (0..replicas).for_each(|idx| self.insert_node(idx, node.clone()))
    }

    fn insert_node(&mut self, replica: usize, node: T) {
        self.ring.ord_insert(self.hash((replica, &node)), node);
    }

    fn hash<K: Hash>(&self, key: K) -> u64 {
        let mut digest = self.hash.build_hasher();
        key.hash(&mut digest);
        digest.finish()
    }

    /// Remove a node from the ring.
    ///
    /// Any keys that were mapped to this node will be uniformly distributed
    /// amongst any remaining nodes.
    ///
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new_with_nodes(10, 0..12);
    /// assert_eq!(12, ring.len());
    /// assert_eq!(10, ring.weight(&3));
    /// ring.remove(&3);
    /// assert_eq!(11, ring.len());
    /// assert_eq!(0, ring.weight(&3));
    /// ```
    pub fn remove(&mut self, node: &T) {
        self.ring.retain(|(_, _node)| node != _node)
    }

    /// Hash the provided key and return a reference to the node responsible
    /// for it.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Returns None if there are no nodes in the ring.
    ///
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new(12);
    /// assert_eq!(None, ring.try_get(b"none"));
    ///
    /// ring.insert(b"hello worldo");
    /// assert_eq!(Some(b"hello worldo"), ring.try_get(42));
    /// ```
    pub fn try_get<K: Hash>(&self, key: K) -> Option<&T> {
        self.ring.find_gte(&self.hash(key))
    }

    /// Hash the provided key and return a reference to the node responsible
    /// for it.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Panics if there are no nodes in the ring.
    /// ```
    /// use consistent_hash_ring::Ring;
    ///
    /// let mut ring = Ring::new_with_nodes(12, 0..3);
    /// assert_eq!(&1, ring.get("ok"));
    /// assert_eq!(&0, ring.get("or"));
    /// assert_eq!(&2, ring.get("another"));
    /// ```
    pub fn get<K: Hash>(&self, key: K) -> &T {
        self.try_get(key).unwrap()
    }
}

#[cfg(test)]
mod consistent_hash_ring_tests {
    extern crate test;

    use super::*;
    use std::collections::HashMap;
    use test::Bencher;

    const TEST_REPLICAS: usize = 4;

    #[test]
    fn remove_insert_is_idempotent() {
        for replicas in 1..=TEST_REPLICAS {
            println!("replicas: {}", replicas);

            let mut ring = Ring::new_with_nodes(replicas, 0..16);

            let x = *ring.get("hello_worldo");
            ring.remove(&x);
            ring.insert(&x);
            let y = *ring.get("hello_worldo");

            assert_eq!(x, y);
        }
    }

    #[test]
    fn is_consistent() {
        for replicas in 1..=TEST_REPLICAS {
            println!("replicas: {}", replicas);

            let ring1 = Ring::new_with_nodes(replicas, vec![0, 1, 2].into_iter());
            let ring2 = Ring::new_with_nodes(replicas, vec![1, 2, 0].into_iter());

            assert_eq!(ring1.get(1), ring2.get(1));
            assert_eq!(ring1.get(2), ring2.get(2));
            assert_eq!(ring1.get(0), ring2.get(0));
        }
    }

    #[test]
    fn try_get_does_not_panic() {
        let ring: Ring<usize> = Ring::new(1);
        assert_eq!(None, ring.try_get("helloworldo"));
    }

    #[test]
    fn removing_nodes_does_not_redistribute_all_keys() {
        for replicas in 1..=TEST_REPLICAS {
            println!("replicas: {}", replicas);

            let mut ring = Ring::new_with_nodes(replicas, 0..4);

            let table = (0..64)
                .map(|x| (x, *ring.get(x)))
                .collect::<HashMap<_, _>>();

            const REMOVED: usize = 2;
            ring.remove(&REMOVED);

            for x in 0..32 {
                let s = table[&x];

                if s != REMOVED {
                    assert_eq!(s, *ring.get(x));
                }
            }
        }
    }

    #[test]
    fn inserting_nodes_does_not_redistribute_all_keys() {
        for replicas in 1..=TEST_REPLICAS {
            println!("replicas: {}", replicas);

            let mut a_ring = Ring::new_with_nodes(replicas, 0..4);
            let mut b_ring = a_ring.clone();

            const A: usize = 42;
            const B: usize = 24;
            a_ring.insert(&A);
            b_ring.insert(&B);

            for x in 0..32 {
                let a = *a_ring.get(x);
                let b = *b_ring.get(x);

                if a != A && b != B {
                    assert_eq!(a, b);
                }
            }
        }
    }

    fn bench_get(b: &mut Bencher, shards: usize) {
        let mut ring = Ring::new(50);

        let buckets: Vec<String> = (0..shards)
            .map(|s| format!("shard-{}", s))
            .inspect(|b| ring.insert(b))
            .collect();

        let mut i = 0;
        b.iter(|| {
            i += 1;
            ring.get(&buckets[i & (shards - 1)]);
        });
    }

    #[bench]
    fn bench_get8(b: &mut Bencher) {
        bench_get(b, 8);
    }

    #[bench]
    fn bench_get32(b: &mut Bencher) {
        bench_get(b, 32);
    }

    #[bench]
    fn bench_get128(b: &mut Bencher) {
        bench_get(b, 128);
    }

    #[bench]
    fn bench_get512(b: &mut Bencher) {
        bench_get(b, 512);
    }
}
