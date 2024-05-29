use std::{
    array::from_fn,
    collections::{BinaryHeap, HashSet, VecDeque},
    hash::Hash,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl Direction {
    pub fn from_component_direction(component: usize, positive: bool) -> Direction {
        match (component, positive) {
            (0, true) => Direction::PosX,
            (0, false) => Direction::NegX,
            (1, true) => Direction::PosY,
            (1, false) => Direction::NegY,
            (2, true) => Direction::PosZ,
            (2, false) => Direction::NegZ,
            _ => panic!("Invalid component direction"),
        }
    }
    pub fn to_offset(&self) -> [i32; 3] {
        match self {
            Direction::PosX => [1, 0, 0],
            Direction::NegX => [-1, 0, 0],
            Direction::PosY => [0, 1, 0],
            Direction::NegY => [0, -1, 0],
            Direction::PosZ => [0, 0, 1],
            Direction::NegZ => [0, 0, -1],
        }
    }

    pub fn component_index(&self) -> usize {
        match self {
            Direction::PosX | Direction::NegX => 0,
            Direction::PosY | Direction::NegY => 1,
            Direction::PosZ | Direction::NegZ => 2,
        }
    }

    pub fn is_positive(&self) -> bool {
        match self {
            Direction::PosX | Direction::PosY | Direction::PosZ => true,
            Direction::NegX | Direction::NegY | Direction::NegZ => false,
        }
    }
}

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

    pub fn keep_if<F>(&mut self, f: F)
    where
        F: Fn(&T) -> bool,
    {
        let mut new_queue = VecDeque::new();
        for item in self.queue.iter() {
            if f(item) {
                new_queue.push_back(item.clone());
            } else {
                self.set.remove(item);
            }
        }
        self.queue = new_queue;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UpdateWithPriority(pub [u32; 3], pub i32);

impl Ord for UpdateWithPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.1.cmp(&other.1)
    }
}

impl PartialOrd for UpdateWithPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct PriorityUpdateQueue {
    queue: BinaryHeap<UpdateWithPriority>,
    set: HashSet<[u32;3]>,
}

impl PriorityUpdateQueue {
    pub fn new() -> PriorityUpdateQueue {
        PriorityUpdateQueue {
            queue: BinaryHeap::new(),
            set: HashSet::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> PriorityUpdateQueue {
        PriorityUpdateQueue {
            queue: BinaryHeap::with_capacity(capacity),
            set: HashSet::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, item: UpdateWithPriority) {
        if self.set.insert(item.0.clone()) {
            self.queue.push(item);
        }
    }

    pub fn pop(&mut self) -> Option<UpdateWithPriority> {
        let item = self.queue.pop();
        if let Some(item) = item {
            self.set.remove(&item.0);
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

    pub fn into_iter(self) -> std::collections::binary_heap::IntoIter<UpdateWithPriority> {
        self.queue.into_iter()
    }

    pub fn keep_if<F>(&mut self, f: F)
    where
        F: Fn(&UpdateWithPriority) -> bool,
    {
        let mut new_queue = BinaryHeap::new();
        for item in self.queue.iter() {
            if f(item) {
                new_queue.push(item.clone());
            } else {
                self.set.remove(&item.0);
            }
        }
        self.queue = new_queue;
    }
}

pub struct VoxelUpdateQueue {
    queue_sets: [[[PriorityUpdateQueue; 2]; 2]; 2],
    active_queue_set: [u32; 3],
}

impl VoxelUpdateQueue {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue_sets: from_fn(|_| from_fn(|_| from_fn(|_| PriorityUpdateQueue::with_capacity(capacity)))),
            active_queue_set: [0, 0, 0],
        }
    }

    pub fn push_all(&mut self, item: [u32; 3]) {
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    self.queue_sets[x][y][z].push(UpdateWithPriority(item, 0));
                }
            }
        }
    }

    pub fn push_with_priority(&mut self, item: UpdateWithPriority) {
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    self.queue_sets[x][y][z].push(item.clone());
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

    fn active_queue_set(&mut self) -> &mut PriorityUpdateQueue {
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

    pub fn pop(&mut self) -> Option<UpdateWithPriority> {
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

    pub fn keep_if<F>(&mut self, f: F)
    where
        F: Fn(&UpdateWithPriority) -> bool,
    {
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    self.queue_sets[x][y][z].keep_if(&f);
                }
            }
        }
    }
}
