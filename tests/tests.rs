use consistent_hash_ring::*;

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
        assert!(ring.remove(&x));
        assert!(ring.insert(x));
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
            .nodes_iter(0..32)
            .build();
        let control = ring.clone();

        const REMOVED: usize = 2;
        assert!(ring.remove(&REMOVED));

        for x in 0..64 {
            let ctl: Vec<_> = control.replicas(x).take(3).collect();
            assert_eq!(*ctl[0], control[x]);

            let real: Vec<_> = ring.replicas(x).take(3).collect();
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
            .nodes_iter(0..4)
            .build();
        let mut y_ring = x_ring.clone();

        const X: usize = 42;
        const Y: usize = 24;
        assert!(x_ring.insert(X));
        assert!(y_ring.insert(Y));

        for v in 0..64 {
            let xs: Vec<_> = x_ring.replicas(v).take(3).collect();
            assert_eq!(*xs[0], x_ring[v]);

            let ys: Vec<_> = y_ring.replicas(v).take(3).collect();
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

    assert!(ring.remove("localhost"));
    assert_eq!(0, ring.len());
    assert_eq!(0, ring.vnodes());
    assert_eq!(None, ring.weight("localhost"));
}

#[test]
fn replicas_are_bounded() {
    let ring = RingBuilder::default().vnodes(12).nodes_iter(0..46).build();
    let reps: Vec<_> = ring.replicas("testkey").collect();
    assert_eq!(reps.len(), ring.len());
}
