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

extern crate fnv;

pub mod collections;
use collections::{first, Map, Set};

use fnv::FnvBuildHasher;
use std::{
    borrow::Borrow,
    cmp,
    hash::{BuildHasher, Hash, Hasher},
    iter::FromIterator,
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
            hasher: self.hasher,
            vnodes: Vec::with_capacity(self.vnodes * self.nodes.len()),
            unique: Vec::with_capacity(self.nodes.len() + self.weighted_nodes.len()),
        };

        let vnodes = self.vnodes;

        self.nodes
            .into_iter()
            .map(|n| (n, vnodes))
            .chain(self.weighted_nodes)
            .for_each(|(n, v)| {
                ring.insert_weight(n, v);
            });

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

impl<T: Hash + Eq + Clone> FromIterator<T> for Ring<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        RingBuilder::default().nodes_iter(iter.into_iter()).build()
    }
}

impl<K: Hash, T: Hash + Eq + Clone, S: BuildHasher> Index<K> for Ring<T, S> {
    type Output = T;

    fn index(&self, index: K) -> &Self::Output {
        self.get(index)
    }
}

impl<T: Hash + Eq + Clone, S: BuildHasher> Extend<T> for Ring<T, S> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();

        let (min, max) = iter.size_hint();
        let n = max.unwrap_or(min);

        self.vnodes.reserve(n * self.n_vnodes);
        self.unique.reserve(n);

        for node in iter {
            self.insert(node);
        }
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
    /// assert!(ring.insert(&()));
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
        self.unique.map_lookup(&self.hash(node)).copied()
    }

    /// Insert a node into the ring with the default vnode count.
    ///
    /// If the provided node is already present in the ring, its vnode count is updated
    /// and `false` is returned.
    ///
    /// O(k * v + (v * log k) + log k + n + log n)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = Ring::default();
    /// assert!(ring.insert("hello worldo"));
    /// assert!(!ring.insert("hello worldo"));
    /// assert_eq!(1, ring.len());
    /// assert_eq!(Some(10), ring.weight("hello worldo"));
    /// ```
    pub fn insert(&mut self, node: T) -> bool {
        self.insert_weight(node, self.n_vnodes)
    }

    /// Insert a node into the ring with the provided number of vnodes.
    ///
    /// This can be used give some nodes more weight than others - nodes with
    /// more vnodes will be selected for larger proportions of keys.
    ///
    /// If the provided node is already present in the ring, its vnode count is updated
    /// and `false` is returned.
    ///
    /// O(k * v + (v * log k) + log k + n + log n)
    ///
    /// ```
    /// use consistent_hash_ring::*;
    ///
    /// let mut ring = Ring::default();
    /// assert!(ring.insert("hello worldo"));
    /// assert!(ring.insert_weight("worldo hello", 9));
    /// assert_eq!(2, ring.len());
    /// assert_eq!(Some(10), ring.weight("hello worldo"));
    /// assert_eq!(Some(9), ring.weight("worldo hello"));
    /// ```
    pub fn insert_weight(&mut self, node: T, vnodes: usize) -> bool {
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

        self.unique.map_insert(node_hash, vnodes).is_none()
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
    /// Returns `true` if the node existed in the ring and was removed.
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
    /// assert!(ring.remove(&3));
    /// assert!(!ring.remove(&3));
    /// assert_eq!(11, ring.len());
    /// assert_eq!(None, ring.weight(&3));
    /// ```
    pub fn remove<Q: ?Sized>(&mut self, node: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.vnodes.retain(|(_, (_node, _))| node != _node.borrow());
        self.unique.map_remove(&self.hash(node)).is_some()
    }

    /// Remove all nodes from the ring.
    ///
    /// O(1)
    pub fn clear(&mut self) {
        self.vnodes.clear();
        self.unique.clear();
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
    /// assert!(ring.insert("hello worldo"));
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
    ///     .nodes_iter(0..12)
    ///     .build();
    ///
    /// let expect = vec![&10, &2, &8];
    ///
    /// assert_eq!(expect, ring.replicas("key").take(3).collect::<Vec<_>>());
    /// assert_eq!(expect[0], ring.get("key"));
    ///
    /// let ring: Ring<()> = RingBuilder::default().build();
    ///
    /// assert_eq!(None, ring.replicas("key").next());
    /// ```
    pub fn replicas<K: Hash>(&self, key: K) -> Candidates<'_, T, S> {
        Candidates {
            inner: self,
            seen: Vec::with_capacity(self.len()),
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
    inner: &'a Ring<T, S>, // the inner ring
    seen: Vec<u64>,        // hashes of nodes we've used
    hash: u64,             // recursive key hash
}

impl<'a, T: Hash + Eq + Clone, S: BuildHasher> Iterator for Candidates<'a, T, S> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.seen.len() >= self.inner.len() {
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

        if self.seen.len() < self.inner.len() {
            self.hash = self.inner.hash(self.hash);
        }

        Some(unsafe { self.inner.get_node_ref(idx) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.inner.len() - self.seen.len();
        (n, Some(n))
    }
}

impl<'a, T: Hash + Eq + Clone, S: BuildHasher> ExactSizeIterator for Candidates<'a, T, S> {
    fn len(&self) -> usize {
        self.inner.len() - self.seen.len()
    }
}
