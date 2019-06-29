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

extern crate fnv;

pub mod collections;
use collections::{Map, Set};

use fnv::FnvBuildHasher;
use std::{
    cmp,
    hash::{BuildHasher, Hash, Hasher},
};

pub struct RingBuilder<T, S = FnvBuildHasher>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    hasher: S,
    vnodes: Option<usize>,
    replicas: Option<usize>,
    nodes: Vec<T>,
    weighted_nodes: Vec<(T, usize)>,
}

impl<T: Hash + Eq + Clone> Default for RingBuilder<T> {
    fn default() -> Self {
        RingBuilder::new(Default::default())
    }
}

impl<T: Hash + Eq + Clone, S: BuildHasher> RingBuilder<T, S> {
    pub fn new(hasher: S) -> Self {
        RingBuilder {
            hasher,
            vnodes: None,
            replicas: None,
            nodes: vec![],
            weighted_nodes: vec![],
        }
    }

    pub fn vnodes(mut self, vnodes: usize) -> Self {
        self.vnodes = Some(vnodes);
        self
    }

    pub fn replicas(mut self, replicas: usize) -> Self {
        self.replicas = Some(replicas);
        self
    }

    pub fn weighted_node(mut self, node: T, vnodes: usize) -> Self {
        self.weighted_nodes.push((node, vnodes));
        self
    }

    pub fn weighted_nodes(mut self, weighted_nodes: &[(T, usize)]) -> Self {
        self.weighted_nodes.extend_from_slice(weighted_nodes);
        self
    }

    pub fn weighted_nodes_iter<I>(mut self, weighted_nodes: I) -> Self
    where
        I: Iterator<Item = (T, usize)>,
    {
        weighted_nodes.for_each(|w_node| self.weighted_nodes.push(w_node));
        self
    }

    pub fn node(mut self, node: T) -> Self {
        self.nodes.push(node);
        self
    }

    pub fn nodes(mut self, nodes: &[T]) -> Self {
        self.nodes.extend_from_slice(nodes);
        self
    }

    pub fn nodes_iter<I>(mut self, nodes: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        nodes.for_each(|node| self.nodes.push(node));
        self
    }

    pub fn build(self) -> Ring<T, S> {
        let vnodes = self.vnodes.unwrap_or(10);

        let mut ring = Ring {
            vnodes,
            replicas: self.replicas.unwrap_or(1),
            ring: Vec::with_capacity(vnodes * self.nodes.len()),
            uniq: Vec::with_capacity(self.nodes.len() + self.weighted_nodes.len()),
            hasher: self.hasher,
        };

        self.nodes.into_iter().for_each(|node| ring.insert(&node));
        self.weighted_nodes
            .into_iter()
            .for_each(|(node, weight)| ring.insert_weight(&node, weight));

        ring
    }
}

/// A consistent hash ring with support for vnodes and key replication.
#[derive(Clone)]
pub struct Ring<T: Hash + Eq + Clone, S = FnvBuildHasher> {
    vnodes: usize,       // default number of vnodes for each node
    replicas: usize,     // default number of replication candidates for each key
    ring: Vec<(u64, T)>, // the ring itself, contains all vnodes
    uniq: Vec<(u64, T)>, // unique nodes for efficient len() implementation
    hasher: S,           // selected build hasher
}

impl<T: Hash + Eq + Clone> Default for Ring<T> {
    fn default() -> Self {
        RingBuilder::default().build()
    }
}

impl<T: Hash + Eq + Clone, S: BuildHasher> Ring<T, S> {
    /// Returns the number of nodes in the ring (not including any duplicate vnodes).
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let ring = RingBuilder::default()
    ///     .nodes_iter(0..32)
    ///     .build();
    /// assert_eq!(32, ring.len());
    /// ```
    pub fn len(&self) -> usize {
        self.uniq.len()
    }

    /// Returns whether or not the ring is empty.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = Ring::default();
    /// assert!(ring.is_empty());
    /// ring.insert(&());
    /// assert!(!ring.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Returns the total number of vnodes in the ring, across all nodes.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let ring = RingBuilder::default()
    ///     .vnodes(4)
    ///     .nodes_iter(0..12)
    ///     .build();
    /// assert_eq!(48, ring.vnodes());
    /// assert_eq!(12, ring.len());
    /// ```
    pub fn vnodes(&self) -> usize {
        self.ring.len()
    }

    /// Returns the number of vnodes for the provided node, or 0 if it does
    /// not exist.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .vnodes(12)
    ///     .nodes_iter(0..4)
    ///     .build();
    /// ring.insert_weight(&42, 9);
    /// assert_eq!(12, ring.weight(&3));
    /// assert_eq!(9, ring.weight(&42));
    /// assert_eq!(0, ring.weight(&24));
    /// ```
    pub fn weight(&self, node: &T) -> usize {
        self.ring.iter().filter(|(_, _node)| _node == node).count()
    }

    /// Insert a node into the ring with the default vnode count.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = Ring::default();
    /// ring.insert(b"hello worldo");
    /// assert_eq!(1, ring.len());
    /// assert_eq!(10, ring.weight(b"hello worldo"));
    /// ```
    pub fn insert(&mut self, node: &T) {
        self.insert_weight(node, self.vnodes)
    }

    /// Insert a node into the ring with the provided number of vnodes.
    ///
    /// This can be used give some nodes more weight than others - nodes
    /// with more vnodes will be selected for larger proportions of keys.
    ///
    /// If the provided node is already present in the ring with a lower
    /// vnode count, the count is updated. If the existing count is higher,
    /// this method does nothing.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = Ring::default();
    /// ring.insert(b"hello worldo");
    /// ring.insert_weight(b"worldo hello", 9);
    /// assert_eq!(2, ring.len());
    /// assert_eq!(10, ring.weight(b"hello worldo"));
    /// assert_eq!(9, ring.weight(b"worldo hello"));
    /// ```
    pub fn insert_weight(&mut self, node: &T, vnodes: usize) {
        for idx in 0..vnodes as u64 {
            let hash = self.hash((idx, node));
            self.ring.map_insert(hash, node.clone());
        }
        self.uniq.map_insert(self.hash(node), node.clone());
    }

    fn hash<K: Hash>(&self, key: K) -> u64 {
        let mut digest = self.hasher.build_hasher();
        key.hash(&mut digest);
        digest.finish()
    }

    /// Remove a node from the ring.
    ///
    /// Any keys that were mapped to this node will be uniformly distributed
    /// amongst any remaining nodes.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .nodes_iter(0..12)
    ///     .build();
    /// assert_eq!(12, ring.len());
    /// assert_eq!(10, ring.weight(&3));
    /// ring.remove(&3);
    /// assert_eq!(11, ring.len());
    /// assert_eq!(0, ring.weight(&3));
    /// ```
    pub fn remove(&mut self, node: &T) {
        self.ring.retain(|(_, _node)| node != _node);
        self.uniq.map_remove(&self.hash(node));
    }

    /// Returns a reference to the first node responsible for the provided key.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Returns None if there are no nodes in the ring.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .vnodes(12)
    ///     .build();
    /// assert_eq!(None, ring.try_first(b"none"));
    ///
    /// ring.insert(b"hello worldo");
    /// assert_eq!(Some(b"hello worldo"), ring.try_first(42));
    /// ```
    pub fn try_first<K: Hash>(&self, key: K) -> Option<&T> {
        self.ring.find_gte(&self.hash(key))
    }

    /// Returns a reference to the first node responsible for the provided key.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Panics if there are no nodes in the ring.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .vnodes(12)
    ///     .nodes_iter(0..1)
    ///     .build();
    ///
    /// assert_eq!(&0, ring.first("by"));
    /// ```
    pub fn first<K: Hash>(&self, key: K) -> &T {
        self.try_first(key).unwrap()
    }

    /// Returns an iterator over replication candidates for the provided key.
    ///
    /// Prefer `Ring::first` instead if only the first node will be used.
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let ring = RingBuilder::default()
    ///     .replicas(3)
    ///     .nodes_iter(0..12)
    ///     .build();
    ///
    /// let expect = vec![&1, &2, &6];
    ///
    /// assert_eq!(expect, ring.get("key").collect::<Vec<_>>());
    /// assert_eq!(expect[0], ring.first("key"));
    /// ```
    pub fn get<'a, K: Hash>(&'a self, key: K) -> Candidates<'a, T, S> {
        Candidates {
            limit: cmp::min(self.replicas, self.len()),
            ring: self,
            seen: Vec::with_capacity(self.replicas),
            hash: self.hash(&key),
        }
    }
}

/// An iterator over replication candidates.
///
/// Constructed by `Ring::get`.
pub struct Candidates<'a, T, S = FnvBuildHasher>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    limit: usize,
    ring: &'a Ring<T, S>,
    seen: Vec<u64>,
    hash: u64,
}

impl<'a, T: Hash + Eq + Clone, S: BuildHasher> Iterator for Candidates<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.seen.len() >= self.limit {
            return None;
        }

        let node = loop {
            let node = self.ring.ring.find_gte(&self.hash)?;
            if self.seen.set_insert(self.ring.hash(node)) {
                break node;
            }
            self.hash = self.ring.hash(self.hash);
        };

        if self.seen.len() < self.limit {
            self.hash = self.ring.hash(self.hash);
        }

        Some(node)
    }
}

#[cfg(test)]
mod consistent_hash_ring_tests {
    extern crate test;

    use super::*;
    use std::collections::HashMap;
    use test::Bencher;

    const TEST_VNODES: usize = 4;

    #[test]
    fn remove_insert_is_idempotent() {
        for vnodes in 1..=TEST_VNODES {
            println!("vnodes: {}", vnodes);

            let mut ring = RingBuilder::default()
                .vnodes(vnodes)
                .nodes_iter(0..16)
                .build();

            let x = *ring.first("hello_worldo");
            ring.remove(&x);
            ring.insert(&x);
            let y = *ring.first("hello_worldo");

            assert_eq!(x, y);
        }
    }

    #[test]
    fn is_consistent() {
        for vnodes in 1..=TEST_VNODES {
            println!("vnodes: {}", vnodes);

            let ring1 = RingBuilder::default()
                .vnodes(vnodes)
                .nodes_iter(vec![0, 1, 2].into_iter())
                .build();
            let ring2 = RingBuilder::default()
                .vnodes(vnodes)
                .nodes_iter(vec![1, 2, 0].into_iter())
                .build();

            assert_eq!(ring1.first(1), ring2.first(1));
            assert_eq!(ring1.first(2), ring2.first(2));
            assert_eq!(ring1.first(0), ring2.first(0));
        }
    }

    #[test]
    fn try_first_does_not_panic() {
        let ring: Ring<usize> = Ring::default();
        assert_eq!(None, ring.try_first("helloworldo"));
    }

    #[test]
    fn removing_nodes_does_not_redistribute_all_keys() {
        for vnodes in 1..=TEST_VNODES {
            println!("vnodes: {}", vnodes);

            let mut ring = RingBuilder::default()
                .vnodes(vnodes)
                .nodes_iter(0..4)
                .build();

            let table = (0..64)
                .map(|x| (x, *ring.first(x)))
                .collect::<HashMap<_, _>>();

            const REMOVED: usize = 2;
            ring.remove(&REMOVED);

            for x in 0..32 {
                let s = table[&x];

                if s != REMOVED {
                    assert_eq!(s, *ring.first(x));
                }
            }
        }
    }

    #[test]
    fn inserting_nodes_does_not_redistribute_all_keys() {
        for vnodes in 1..=TEST_VNODES {
            println!("vnodes: {}", vnodes);

            let mut a_ring = RingBuilder::default()
                .vnodes(vnodes)
                .nodes_iter(0..4)
                .build();
            let mut b_ring = a_ring.clone();

            const A: usize = 42;
            const B: usize = 24;
            a_ring.insert(&A);
            b_ring.insert(&B);

            for x in 0..32 {
                let a = *a_ring.first(x);
                let b = *b_ring.first(x);

                if a != A && b != B {
                    assert_eq!(a, b);
                }
            }
        }
    }

    fn bench_get32(b: &mut Bencher, replicas: usize) {
        let mut ring = RingBuilder::default().vnodes(50).replicas(replicas).build();

        let buckets: Vec<String> = (0..32)
            .map(|s| format!("shard-{}", s))
            .inspect(|b| ring.insert(b))
            .collect();

        let mut i = 0;
        b.iter(|| {
            i += 1;
            ring.get(&buckets[i & 31]).for_each(|_| ());
        });
    }

    #[bench]
    fn bench_get32_1(b: &mut Bencher) {
        bench_get32(b, 1);
    }

    #[bench]
    fn bench_get32_2(b: &mut Bencher) {
        bench_get32(b, 2);
    }

    #[bench]
    fn bench_get32_4(b: &mut Bencher) {
        bench_get32(b, 4);
    }

    #[bench]
    fn bench_get32_8(b: &mut Bencher) {
        bench_get32(b, 8);
    }

    fn bench_first(b: &mut Bencher, shards: usize) {
        let mut ring = RingBuilder::default().vnodes(50).build();

        let buckets: Vec<String> = (0..shards)
            .map(|s| format!("shard-{}", s))
            .inspect(|b| ring.insert(b))
            .collect();

        let mut i = 0;
        b.iter(|| {
            i += 1;
            ring.first(&buckets[i & (shards - 1)]);
        });
    }

    #[bench]
    fn bench_first_1_8(b: &mut Bencher) {
        bench_first(b, 8);
    }

    #[bench]
    fn bench_first_2_32(b: &mut Bencher) {
        bench_first(b, 32);
    }

    #[bench]
    fn bench_first_3_128(b: &mut Bencher) {
        bench_first(b, 128);
    }

    #[bench]
    fn bench_first_4_512(b: &mut Bencher) {
        bench_first(b, 512);
    }
}
