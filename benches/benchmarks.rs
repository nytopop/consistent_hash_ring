#![feature(test)]
use consistent_hash_ring::*;
extern crate test;
use test::Bencher;

fn bench_replicas32(b: &mut Bencher, replicas: usize) {
    let mut ring = RingBuilder::default().vnodes(50).build();

    let buckets: Vec<String> = (0..32)
        .map(|s| format!("shard-{}", s))
        .inspect(|b| {
            ring.insert(b.clone());
        })
        .collect();

    let mut i = 0;
    b.iter(|| {
        i += 1;
        ring.replicas(&buckets[i & 31])
            .take(replicas)
            .for_each(|_| ());
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
        .inspect(|b| {
            ring.insert(b.clone());
        })
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
