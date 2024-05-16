use std::{
    array::from_fn,
    collections::{HashSet, VecDeque},
    hash::Hash,
};

pub struct QueueSet<T: Clone + Eq + PartialEq + Hash> {
    queue: VecDeque<T>,
    set: HashSet<T>,
}

impl<T: Clone + Eq + PartialEq + Hash> QueueSet<T> {
    pub fn new() -> QueueSet<T> {
        QueueSet {
            queue: VecDeque::new(),
            set: HashSet::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> QueueSet<T> {
        QueueSet {
            queue: VecDeque::with_capacity(capacity),
            set: HashSet::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, item: T) {
        if self.set.insert(item.clone()) {
            self.queue.push_back(item);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let item = self.queue.pop_front();
        if let Some(item) = item {
            self.set.remove(&item);
            Some(item)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn into_iter(self) -> std::collections::vec_deque::IntoIter<T> {
        self.queue.into_iter()
    }
}

pub struct VoxelUpdateQueue {
    queue_sets: [[[QueueSet<[u32; 3]>; 2]; 2]; 2],
    active_queue_set: [u32; 3],
}

impl VoxelUpdateQueue {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue_sets: from_fn(|_| from_fn(|_| from_fn(|_| QueueSet::with_capacity(capacity)))),
            active_queue_set: [0, 0, 0],
        }
    }

    pub fn push_all(&mut self, item: [u32; 3]) {
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    self.queue_sets[x][y][z].push(item);
                }
            }
        }
    }

    pub fn next_queue_set(&mut self) {
        self.active_queue_set[0] = (self.active_queue_set[0] + 1) % 2;
        if self.active_queue_set[0] == 0 {
            self.active_queue_set[1] = (self.active_queue_set[1] + 1) % 2;
            if self.active_queue_set[1] == 0 {
                self.active_queue_set[2] = (self.active_queue_set[2] + 1) % 2;
            }
        }
    }

    fn active_queue_set(&mut self) -> &mut QueueSet<[u32; 3]> {
        &mut self.queue_sets[self.active_queue_set[0] as usize][self.active_queue_set[1] as usize]
            [self.active_queue_set[2] as usize]
    }

    pub fn queue_set_idx(&self) -> [u32; 3] {
        self.active_queue_set
    }

    pub fn swap_queue_set(&mut self) {
        loop {
            self.next_queue_set();
            if !self.active_queue_set().is_empty() {
                break;
            }
        }
    }

    pub fn pop(&mut self) -> Option<[u32; 3]> {
        self.active_queue_set().pop()
    }

    pub fn len(&self) -> usize {
        self.queue_sets
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| y.iter().map(|z| z.len()).sum::<usize>())
                    .sum::<usize>()
            })
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.queue_sets
            .iter()
            .all(|x| x.iter().all(|y| y.iter().all(|z| z.is_empty())))
    }
}
