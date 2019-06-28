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
    hash::{BuildHasher, BuildHasherDefault, Hash, Hasher},
    rc::Rc,
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
        self.0 as u64
    }
}

type IEEE = BuildHasherDefault<DigestIEEE>;

/// A consistent hash ring with variable weight nodes.
#[derive(Clone)]
pub struct Ring<T: Hash + Eq, S = IEEE> {
    replicas: usize,          // default number of replicas for each node
    ring: VecMap<u64, Rc<T>>, // the ring itself
    hash: S,                  // selected build hasher
}

impl<T: Hash + Eq> Default for Ring<T> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T: Hash + Eq> Ring<T> {
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
        nodes.for_each(|node| ring.insert(node));
        ring
    }
}

impl<T: Hash + Eq, S: BuildHasher> Ring<T, S> {
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
        nodes.for_each(|node| ring.insert(node));
        ring
    }

    fn hash<K: Hash>(&self, key: K) -> u64 {
        let mut digest = self.hash.build_hasher();
        key.hash(&mut digest);
        digest.finish()
    }

    fn insert_node(&mut self, replica: (usize, Rc<T>)) {
        self.ring.ord_insert(self.hash(&replica), replica.1);
    }

    /// Insert a node into the ring with the default replica count.
    pub fn insert(&mut self, node: T) {
        let node = Rc::new(node);
        (0..self.replicas)
            .map(|idx| (idx, node.clone()))
            .for_each(|replica| self.insert_node(replica));
    }

    /// Insert a node into the ring with a variable amount of replicas.
    ///
    /// This can be used give some nodes more weight than others - nodes
    /// with higher replica counts will tend to be selected for larger
    /// proportions of keys.
    pub fn insert_weight(&mut self, node: T, replicas: usize) {
        let node = Rc::new(node);
        (0..replicas)
            .map(|idx| (idx, node.clone()))
            .for_each(|replica| self.insert_node(replica));
    }

    /// Remove a node from the ring.
    pub fn remove(&mut self, node: &T) {
        self.ring.retain(|(_, _node)| *node != **_node);
    }

    /// Hash the provided key and return a reference to the first node
    /// responsible for it.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Returns None if there are no nodes in the ring.
    pub fn try_get<K: Hash>(&self, key: K) -> Option<Rc<T>> {
        self.ring.find_gte(&self.hash(key)).cloned()
    }

    /// Hash the provided key and return a reference to the first node
    /// responsible for it.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Panics if there are no nodes in the ring.
    pub fn get<K: Hash>(&self, key: K) -> Rc<T> {
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
            ring.insert(x);
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
            a_ring.insert(A);
            b_ring.insert(B);

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
            .inspect(|b| ring.insert(b.to_owned()))
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
