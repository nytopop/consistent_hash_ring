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
#![feature(test)]

extern crate fnv;

pub mod collections;
use collections::{first, Map, Set};

use fnv::FnvBuildHasher;
use std::{
    borrow::Borrow,
    cmp,
    hash::{BuildHasher, Hash, Hasher},
    ops::Index,
};

/// A builder for `Ring`.
#[derive(Clone)]
pub struct RingBuilder<T, S = FnvBuildHasher>
where
    T: Hash + Eq + Clone,
    S: BuildHasher,
{
    hasher: S,
    vnodes: usize,
    replicas: usize,
    nodes: Vec<T>,
    weighted_nodes: Vec<(T, usize)>,
}

impl<T: Hash + Eq + Clone> Default for RingBuilder<T> {
    fn default() -> Self {
        RingBuilder::new(Default::default())
    }
}

impl<T: Hash + Eq + Clone, S: BuildHasher> RingBuilder<T, S> {
    /// Returns a new `RingBuilder` with the specified `BuildHasher`.
    pub fn new(hasher: S) -> Self {
        RingBuilder {
            hasher,
            vnodes: 10,
            replicas: 1,
            nodes: vec![],
            weighted_nodes: vec![],
        }
    }

    /// Specifies the number of vnodes for each node.
    ///
    /// The default is 10.
    pub fn vnodes(mut self, vnodes: usize) -> Self {
        self.vnodes = cmp::max(1, vnodes);
        self
    }

    /// Specifies the number of replicas for each key.
    ///
    /// The default is 1.
    pub fn replicas(mut self, replicas: usize) -> Self {
        self.replicas = cmp::max(1, replicas);
        self
    }

    /// Ensure that the built `Ring` contains node with the specified number of
    /// vnodes (regardless of the default vnode count).
    pub fn weighted_node(mut self, node: T, vnodes: usize) -> Self {
        self.weighted_nodes.push((node, vnodes));
        self
    }

    /// Ensure that the built `Ring` contains all nodes in the provided slice with
    /// the associated number of vnodes (regardless of the default vnode count).
    pub fn weighted_nodes(mut self, weighted_nodes: &[(T, usize)]) -> Self {
        self.weighted_nodes.extend_from_slice(weighted_nodes);
        self
    }

    /// Ensure that the built `Ring` contains all nodes in the provided iterator with
    /// the associated number of vnodes (regardless of the default vnode count).
    pub fn weighted_nodes_iter<I>(mut self, weighted_nodes: I) -> Self
    where
        I: Iterator<Item = (T, usize)>,
    {
        weighted_nodes.for_each(|w_node| self.weighted_nodes.push(w_node));
        self
    }

    /// Ensure that the built `Ring` contains the provided node.
    pub fn node(mut self, node: T) -> Self {
        self.nodes.push(node);
        self
    }

    /// Ensure that the built `Ring` contains all nodes in the provided slice.
    pub fn nodes(mut self, nodes: &[T]) -> Self {
        self.nodes.extend_from_slice(nodes);
        self
    }

    /// Ensure that the built `Ring` contains all nodes in the provided iterator.
    pub fn nodes_iter<I>(mut self, nodes: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        nodes.for_each(|node| self.nodes.push(node));
        self
    }

    /// Build the `Ring`.
    pub fn build(self) -> Ring<T, S> {
        let mut ring = Ring {
            n_vnodes: self.vnodes,
            replicas: self.replicas,
            hasher: self.hasher,
            vnodes: Vec::with_capacity(self.vnodes * self.nodes.len()),
            unique: Vec::with_capacity(self.nodes.len() + self.weighted_nodes.len()),
        };

        let vnodes = self.vnodes;

        self.nodes
            .into_iter()
            .map(|n| (n, vnodes))
            .chain(self.weighted_nodes)
            .for_each(|(n, v)| ring.insert_weight(n, v));

        ring
    }
}

/// A consistent hash ring with support for virtual nodes and key replication.
///
/// ## How it works
/// Typical hash ring construction. See
/// [wikipedia](https://en.wikipedia.org/wiki/Consistent_hashing#Technique).
///
/// Nodes are mapped onto locations on a finite ring using a hash function. Keys are
/// likewise hashed and mapped onto the same ring; the responsible node for a given
/// key is the first node whose hash is greater than or equal to the key's hash.
///
/// ## Virtual Nodes
/// Because node locations on the ring are only uniformly distributed when looking at
/// the ring containing all possible nodes, we map each node to multiple vnodes to
/// improve the statistical properties of key distribution. The probability of a node
/// being selected for any given key scales linearly with the proportion of assigned
/// vnodes relative to other nodes in the ring.
///
/// ## Replication
/// Keys may optionally be mapped to a set of disjoint vnodes for fault tolerance or
/// high availability purposes. Each additional replica is located by hashing keys an
/// extra round; the hash is then mapped in the same manner as a single key. If any
/// duplicate nodes arise during replica resolution, a circular scan starting at the
/// index of the offending vnode is used to select the next vnode.
///
/// ## Performance
/// Mutable and immutable operations on the ring tend toward linear and sublinear
/// complexity respectively.
///
/// The figures given below assume that:
///
/// * n = total nodes
/// * k = total vnodes
/// * v = vnodes per node
/// * r = replica count
#[derive(Clone)]
pub struct Ring<T: Hash + Eq + Clone, S = FnvBuildHasher> {
    n_vnodes: usize, // # of vnodes for each node (default)
    replicas: usize, // # of replication candidates for each key
    hasher: S,       // selected build hasher

    // kv map for vnodes. used for most queries.
    //
    //           |----------- hash(hash(...(hash(node))))
    //           |     |----- node
    //           |     |  |-- hash(node)
    vnodes: Vec<(u64, (T, u64))>,

    // kv map for unique nodes. mainly used for efficient
    // implementations of len() and weight().
    //
    //           |------- hash(node)
    //           |    |-- n_vnodes
    unique: Vec<(u64, usize)>,
}

impl<T: Hash + Eq + Clone> Default for Ring<T> {
    fn default() -> Self {
        RingBuilder::default().build()
    }
}

impl<K: Hash, T: Hash + Eq + Clone, S: BuildHasher> Index<K> for Ring<T, S> {
    type Output = T;

    fn index(&self, index: K) -> &Self::Output {
        self.get(index)
    }
}

impl<T: Hash + Eq + Clone, S: BuildHasher> Ring<T, S> {
    /// Returns the number of nodes in the ring (not including any duplicate vnodes).
    ///
    /// O(1)
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
        self.unique.len()
    }

    /// Returns whether or not the ring is empty.
    ///
    /// O(1)
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
        self.vnodes.is_empty()
    }

    /// Returns the total number of vnodes in the ring, across all nodes.
    ///
    /// O(1)
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
        self.vnodes.len()
    }

    /// Returns the number of vnodes for the provided node, or `None` if it does
    /// not exist.
    ///
    /// O(log n)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .vnodes(12)
    ///     .nodes_iter(0..4)
    ///     .weighted_node(42, 9)
    ///     .build();
    ///
    /// assert_eq!(Some(12), ring.weight(&3));
    /// assert_eq!(Some(9), ring.weight(&42));
    /// assert_eq!(None, ring.weight(&24));
    /// ```
    pub fn weight<Q: ?Sized>(&self, node: &Q) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.unique.map_lookup(&self.hash(node)).map(|w| *w)
    }

    /// Insert a node into the ring with the default vnode count.
    ///
    /// O(k * v + (v * log k) + log k + n + log n)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = Ring::default();
    /// ring.insert("hello worldo");
    /// assert_eq!(1, ring.len());
    /// assert_eq!(Some(10), ring.weight("hello worldo"));
    /// ```
    pub fn insert(&mut self, node: T) {
        self.insert_weight(node, self.n_vnodes)
    }

    /// Insert a node into the ring with the provided number of vnodes.
    ///
    /// This can be used give some nodes more weight than others - nodes with
    /// more vnodes will be selected for larger proportions of keys.
    ///
    /// If the provided node is already present in the ring, the count is updated.
    ///
    /// O(k * v + (v * log k) + log k + n + log n)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = Ring::default();
    /// ring.insert("hello worldo");
    /// ring.insert_weight("worldo hello", 9);
    /// assert_eq!(2, ring.len());
    /// assert_eq!(Some(10), ring.weight("hello worldo"));
    /// assert_eq!(Some(9), ring.weight("worldo hello"));
    /// ```
    pub fn insert_weight(&mut self, node: T, vnodes: usize) {
        let node_hash = self.hash(&node);

        let mut hash = node_hash;
        // insert new vnodes
        for _ in 0..vnodes.saturating_sub(1) {
            self.vnodes.map_insert(hash, (node.clone(), node_hash));
            hash = self.hash(hash);
        }
        if vnodes > 0 {
            self.vnodes.map_insert(hash, (node, node_hash));
            hash = self.hash(hash);
        }

        // remove old vnodes (if any are present)
        while self.vnodes.map_remove(&hash).is_some() {
            hash = self.hash(hash);
        }

        self.unique.map_insert(node_hash, vnodes);
    }

    // Hash the provided key.
    fn hash<K: Hash>(&self, key: K) -> u64 {
        let mut digest = self.hasher.build_hasher();
        key.hash(&mut digest);
        digest.finish()
    }

    /// Remove a node from the ring.
    ///
    /// Any keys that were mapped to this node will be uniformly distributed
    /// amongst nearby nodes.
    ///
    /// O(k + n + log n)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .nodes_iter(0..12)
    ///     .build();
    /// assert_eq!(12, ring.len());
    /// assert_eq!(Some(10), ring.weight(&3));
    /// ring.remove(&3);
    /// assert_eq!(11, ring.len());
    /// assert_eq!(None, ring.weight(&3));
    /// ```
    pub fn remove<Q: ?Sized>(&mut self, node: &Q)
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.vnodes.retain(|(_, (_node, _))| node != _node.borrow());
        self.unique.map_remove(&self.hash(node));
    }

    /// Returns a reference to the first node responsible for the provided key.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Returns `None` if there are no nodes in the ring.
    ///
    /// O(log k)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .vnodes(12)
    ///     .build();
    /// assert_eq!(None, ring.try_get("none"));
    ///
    /// ring.insert("hello worldo");
    /// assert_eq!(Some(&"hello worldo"), ring.try_get(42));
    /// ```
    pub fn try_get<K: Hash>(&self, key: K) -> Option<&T> {
        self.vnodes.find_gte(&self.hash(key)).map(first)
    }

    /// Returns a reference to the first node responsible for the provided key.
    ///
    /// Any key type may be used so long as it is Hash.
    ///
    /// Panics if there are no nodes in the ring.
    ///
    /// O(log k)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = RingBuilder::default()
    ///     .vnodes(12)
    ///     .nodes_iter(0..1)
    ///     .build();
    ///
    /// assert_eq!(&0, ring.get("by"));
    /// ```
    pub fn get<K: Hash>(&self, key: K) -> &T {
        self.try_get(key).unwrap()
    }

    /// Returns an iterator over replication candidates for the provided key.
    ///
    /// Prefer `Ring::get` instead if only the first node will be used.
    ///
    /// O(r * (r + log r + k + log k))
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let ring = RingBuilder::default()
    ///     .replicas(3)
    ///     .nodes_iter(0..12)
    ///     .build();
    ///
    /// let expect = vec![&10, &2, &8];
    ///
    /// assert_eq!(expect, ring.replicas("key").collect::<Vec<_>>());
    /// assert_eq!(expect[0], ring.get("key"));
    ///
    /// let ring: Ring<()> = RingBuilder::default()
    ///     .replicas(3)
    ///     .build();
    ///
    /// assert_eq!(None, ring.replicas("key").next());
    /// ```
    pub fn replicas<'a, K: Hash>(&'a self, key: K) -> Candidates<'a, T, S> {
        Candidates {
            limit: cmp::min(self.replicas, self.len()),
            inner: self,
            seen: Vec::with_capacity(self.replicas),
            hash: self.hash(&key),
        }
    }

    // Get the root hash of the node at the provided vnode index.
    unsafe fn get_root_hash(&self, vnode_idx: usize) -> u64 {
        (self.vnodes.get_unchecked(vnode_idx).1).1
    }

    // Get a reference to the node at the provided vnode index.
    unsafe fn get_node_ref(&self, vnode_idx: usize) -> &T {
        &(self.vnodes.get_unchecked(vnode_idx).1).0
    }
}

/// An iterator over replication candidates.
///
/// Constructed by `Ring::replicas`.
pub struct Candidates<'a, T: Hash + Eq + Clone, S = FnvBuildHasher> {
    limit: usize,          // # of replicas to return
    inner: &'a Ring<T, S>, // the inner ring
    seen: Vec<u64>,        // hashes of nodes we've used
    hash: u64,             // recursive key hash
}

impl<'a, T: Hash + Eq + Clone, S: BuildHasher> Iterator for Candidates<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.seen.len() >= self.limit {
            return None;
        }

        let checked = |i| match self.inner.vnodes.len() {
            n if n == 0 => None,
            n if n == i => Some(0),
            _ => Some(i),
        };

        let mut idx = (self.inner.vnodes)
            .binary_search_by_key(&&self.hash, first)
            .map_or_else(checked, Some)?;

        while !self
            .seen
            .set_insert(unsafe { self.inner.get_root_hash(idx) })
        {
            if idx < self.inner.vnodes.len() - 1 {
                idx += 1;
            } else {
                idx = 0;
            }
        }

        if self.seen.len() < self.limit {
            self.hash = self.inner.hash(self.hash);
        }

        Some(unsafe { self.inner.get_node_ref(idx) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.limit - self.seen.len();
        (n, Some(n))
    }
}

#[cfg(test)]
mod consistent_hash_ring_tests {
    extern crate test;

    use super::*;
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

            let x = ring["hello_worldo"];
            ring.remove(&x);
            ring.insert(x);
            let y = ring["hello_worldo"];

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

            (0..32).for_each(|i| assert_eq!(ring1[i], ring2[i]));
        }
    }

    #[test]
    fn try_get_does_not_panic() {
        let ring: Ring<usize> = Ring::default();
        assert_eq!(None, ring.try_get("helloworldo"));
    }

    #[test]
    fn removing_nodes_does_not_redistribute_all_replicas() {
        for vnodes in 1..=TEST_VNODES {
            println!("vnodes: {}", vnodes);

            let mut ring = RingBuilder::default()
                .vnodes(vnodes)
                .replicas(3)
                .nodes_iter(0..32)
                .build();
            let control = ring.clone();

            const REMOVED: usize = 2;
            ring.remove(&REMOVED);

            for x in 0..64 {
                let ctl: Vec<_> = control.replicas(x).collect();
                assert_eq!(*ctl[0], control[x]);

                let real: Vec<_> = ring.replicas(x).collect();
                assert_eq!(*real[0], ring[x]);

                if !ctl.contains(&&REMOVED) {
                    assert_eq!(ctl, real);
                }
            }
        }
    }

    #[test]
    fn inserting_nodes_does_not_redistribute_all_replicas() {
        for vnodes in 1..=TEST_VNODES {
            println!("vnodes: {}", vnodes);

            let mut x_ring = RingBuilder::default()
                .vnodes(vnodes)
                .replicas(3)
                .nodes_iter(0..4)
                .build();
            let mut y_ring = x_ring.clone();

            const X: usize = 42;
            const Y: usize = 24;
            x_ring.insert(X);
            y_ring.insert(Y);

            for v in 0..64 {
                let xs: Vec<_> = x_ring.replicas(v).collect();
                assert_eq!(*xs[0], x_ring[v]);

                let ys: Vec<_> = y_ring.replicas(v).collect();
                assert_eq!(*ys[0], y_ring[v]);

                if !xs.contains(&&X) && !ys.contains(&&Y) {
                    assert_eq!(xs, ys);
                }
            }
        }
    }

    #[test]
    fn remove_borrowed_node() {
        let mut ring = RingBuilder::default()
            .vnodes(10)
            .node("localhost".to_owned())
            .build();

        assert_eq!(1, ring.len());
        assert_eq!(10, ring.vnodes());
        assert_eq!(Some(10), ring.weight("localhost"));

        ring.remove("localhost");
        assert_eq!(0, ring.len());
        assert_eq!(0, ring.vnodes());
        assert_eq!(None, ring.weight("localhost"));
    }

    fn bench_replicas32(b: &mut Bencher, replicas: usize) {
        let mut ring = RingBuilder::default().vnodes(50).replicas(replicas).build();

        let buckets: Vec<String> = (0..32)
            .map(|s| format!("shard-{}", s))
            .inspect(|b| ring.insert(b.clone()))
            .collect();

        let mut i = 0;
        b.iter(|| {
            i += 1;
            ring.replicas(&buckets[i & 31]).for_each(|_| ());
        });
    }

    #[bench]
    fn bench_replicas32_1(b: &mut Bencher) {
        bench_replicas32(b, 1);
    }

    #[bench]
    fn bench_replicas32_2(b: &mut Bencher) {
        bench_replicas32(b, 2);
    }

    #[bench]
    fn bench_replicas32_3(b: &mut Bencher) {
        bench_replicas32(b, 3);
    }

    #[bench]
    fn bench_replicas32_4(b: &mut Bencher) {
        bench_replicas32(b, 4);
    }

    fn bench_get(b: &mut Bencher, shards: usize) {
        let mut ring = RingBuilder::default().vnodes(50).build();

        let buckets: Vec<String> = (0..shards)
            .map(|s| format!("shard-{}", s))
            .inspect(|b| ring.insert(b.clone()))
            .collect();

        let mut i = 0;
        b.iter(|| {
            i += 1;
            ring.get(&buckets[i & (shards - 1)]);
        });
    }

    #[bench]
    fn bench_get_a_8(b: &mut Bencher) {
        bench_get(b, 8);
    }

    #[bench]
    fn bench_get_b_32(b: &mut Bencher) {
        bench_get(b, 32);
    }

    #[bench]
    fn bench_get_c_128(b: &mut Bencher) {
        bench_get(b, 128);
    }

    #[bench]
    fn bench_get_d_512(b: &mut Bencher) {
        bench_get(b, 512);
    }
}
