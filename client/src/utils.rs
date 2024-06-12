use std::{
    array::from_fn,
    collections::{HashMap, HashSet, VecDeque},
    hash::Hash,
};
use egui_winit_vulkano::egui::{self};
use priority_queue::PriorityQueue;
use winit::event::VirtualKeyCode;
use ahash::RandomState;

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

pub struct QueueMap<K, V>
where
    K: Clone + Eq + PartialEq + Hash,
    V: Clone,
{
    queue: VecDeque<K>,
    map: HashMap<K, V>,
    keep_value: Box<dyn Fn(V, V) -> V>,
}

impl<K, V> QueueMap<K, V>
where
    K: Clone + Eq + PartialEq + Hash,
    V: Clone,
{
    pub fn new<F>(keep_value: F) -> QueueMap<K, V>
    where
        F: 'static + Fn(V, V) -> V,
    {
        QueueMap {
            queue: VecDeque::new(),
            map: HashMap::new(),
            keep_value: Box::new(keep_value),
        }
    }

    pub fn with_capacity<F>(capacity: usize, keep_value: F) -> QueueMap<K, V>
    where
        F: 'static + Fn(V, V) -> V,
    {
        QueueMap {
            queue: VecDeque::with_capacity(capacity),
            map: HashMap::with_capacity(capacity),
            keep_value: Box::new(keep_value),
        }
    }

    pub fn push(&mut self, key: K, value: V) -> bool {
        if let Some(existing_value) = self.map.get_mut(&key) {
            *existing_value = (self.keep_value)(existing_value.clone(), value);
            false // Key already exists, updated with the chosen value
        } else {
            self.queue.push_back(key.clone());
            self.map.insert(key, value);
            true // Key did not exist, added to queue and map
        }
    }

    pub fn pop(&mut self) -> Option<(K, V)> {
        if let Some(key) = self.queue.pop_front() {
            if let Some(value) = self.map.remove(&key) {
                return Some((key, value));
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn into_iter(self) -> std::collections::vec_deque::IntoIter<K> {
        self.queue.into_iter()
    }

    pub fn keep_if<P>(&mut self, predicate: P)
    where
        P: Fn(&K, &V) -> bool,
    {
        let mut new_queue = VecDeque::new();
        for key in self.queue.iter() {
            if let Some(value) = self.map.get(key) {
                if predicate(key, value) {
                    new_queue.push_back(key.clone());
                } else {
                    self.map.remove(key);
                }
            }
        }
        self.queue = new_queue;
    }

    pub fn extend(&mut self, delayed_writes: Vec<(K, V)>) {
        for (key, value) in delayed_writes {
            self.push(key, value);
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

    pub fn extend(&mut self, delayed_writes: Vec<T>) {
        for item in delayed_writes {
            self.push(item);
        }
    }
}

pub struct VoxelUpdateQueue {
    queue_sets: [[[PriorityQueue<[u32; 3], i32, RandomState>; 2]; 2]; 2],
    active_queue_set: [u32; 3],
}

impl VoxelUpdateQueue {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue_sets: from_fn(|_| {
                from_fn(|_| from_fn(|_| PriorityQueue::with_capacity_and_hasher(capacity, RandomState::with_seed(0))))
            }),
            active_queue_set: [0, 0, 0],
        }
    }

    pub fn push_all(&mut self, item: [u32; 3]) {
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    self.queue_sets[x][y][z].push_increase(item, 0);
                }
            }
        }
    }

    pub fn push_with_priority(&mut self, item: [u32; 3], priority: i32) {
        for x in 0..2 {
            for y in 0..2 {
                for z in 0..2 {
                    self.queue_sets[x][y][z].push_increase(item.clone(), priority);
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

    fn active_queue_set(&mut self) -> &mut PriorityQueue<[u32; 3], i32, RandomState> {
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

    pub fn pop(&mut self) -> Option<([u32; 3], i32)> {
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

pub fn translate_egui_key_code(key: egui::Key) -> VirtualKeyCode {
    match key {
        egui::Key::ArrowDown => VirtualKeyCode::Down,
        egui::Key::ArrowLeft => VirtualKeyCode::Left,
        egui::Key::ArrowRight => VirtualKeyCode::Right,
        egui::Key::ArrowUp => VirtualKeyCode::Up,

        egui::Key::Escape => VirtualKeyCode::Escape,
        egui::Key::Tab => VirtualKeyCode::Tab,
        egui::Key::Backspace => VirtualKeyCode::Back,
        egui::Key::Enter => VirtualKeyCode::Return,
        egui::Key::Space => VirtualKeyCode::Space,

        egui::Key::Insert => VirtualKeyCode::Insert,
        egui::Key::Delete => VirtualKeyCode::Delete,
        egui::Key::Home => VirtualKeyCode::Home,
        egui::Key::End => VirtualKeyCode::End,
        egui::Key::PageUp => VirtualKeyCode::PageUp,
        egui::Key::PageDown => VirtualKeyCode::PageDown,

        egui::Key::Minus => VirtualKeyCode::Minus,
        // Using Mac the key with the Plus sign on it is reported as the Equals key
        // (with both English and Swedish keyboard).
        egui::Key::PlusEquals => VirtualKeyCode::Equals,

        egui::Key::Num0 => VirtualKeyCode::Key0,
        egui::Key::Num1 => VirtualKeyCode::Key1,
        egui::Key::Num2 => VirtualKeyCode::Key2,
        egui::Key::Num3 => VirtualKeyCode::Key3,
        egui::Key::Num4 => VirtualKeyCode::Key4,
        egui::Key::Num5 => VirtualKeyCode::Key5,
        egui::Key::Num6 => VirtualKeyCode::Key6,
        egui::Key::Num7 => VirtualKeyCode::Key7,
        egui::Key::Num8 => VirtualKeyCode::Key8,
        egui::Key::Num9 => VirtualKeyCode::Key9,

        egui::Key::A => VirtualKeyCode::A,
        egui::Key::B => VirtualKeyCode::B,
        egui::Key::C => VirtualKeyCode::C,
        egui::Key::D => VirtualKeyCode::D,
        egui::Key::E => VirtualKeyCode::E,
        egui::Key::F => VirtualKeyCode::F,
        egui::Key::G => VirtualKeyCode::G,
        egui::Key::H => VirtualKeyCode::H,
        egui::Key::I => VirtualKeyCode::I,
        egui::Key::J => VirtualKeyCode::J,
        egui::Key::K => VirtualKeyCode::K,
        egui::Key::L => VirtualKeyCode::L,
        egui::Key::M => VirtualKeyCode::M,
        egui::Key::N => VirtualKeyCode::N,
        egui::Key::O => VirtualKeyCode::O,
        egui::Key::P => VirtualKeyCode::P,
        egui::Key::Q => VirtualKeyCode::Q,
        egui::Key::R => VirtualKeyCode::R,
        egui::Key::S => VirtualKeyCode::S,
        egui::Key::T => VirtualKeyCode::T,
        egui::Key::U => VirtualKeyCode::U,
        egui::Key::V => VirtualKeyCode::V,
        egui::Key::W => VirtualKeyCode::W,
        egui::Key::X => VirtualKeyCode::X,
        egui::Key::Y => VirtualKeyCode::Y,
        egui::Key::Z => VirtualKeyCode::Z,

        egui::Key::F1 => VirtualKeyCode::F1,
        egui::Key::F2 => VirtualKeyCode::F2,
        egui::Key::F3 => VirtualKeyCode::F3,
        egui::Key::F4 => VirtualKeyCode::F4,
        egui::Key::F5 => VirtualKeyCode::F5,
        egui::Key::F6 => VirtualKeyCode::F6,
        egui::Key::F7 => VirtualKeyCode::F7,
        egui::Key::F8 => VirtualKeyCode::F8,
        egui::Key::F9 => VirtualKeyCode::F9,
        egui::Key::F10 => VirtualKeyCode::F10,
        egui::Key::F11 => VirtualKeyCode::F11,
        egui::Key::F12 => VirtualKeyCode::F12,
        egui::Key::F13 => VirtualKeyCode::F13,
        egui::Key::F14 => VirtualKeyCode::F14,
        egui::Key::F15 => VirtualKeyCode::F15,
        egui::Key::F16 => VirtualKeyCode::F16,
        egui::Key::F17 => VirtualKeyCode::F17,
        egui::Key::F18 => VirtualKeyCode::F18,
        egui::Key::F19 => VirtualKeyCode::F19,
        egui::Key::F20 => VirtualKeyCode::F20,
    }
}

pub fn translate_egui_pointer_button(button: egui::PointerButton) -> winit::event::MouseButton {
    match button {
        egui::PointerButton::Primary => winit::event::MouseButton::Left,
        egui::PointerButton::Secondary => winit::event::MouseButton::Right,
        egui::PointerButton::Middle => winit::event::MouseButton::Middle,
        egui::PointerButton::Extra1 => winit::event::MouseButton::Other(1),
        egui::PointerButton::Extra2 => winit::event::MouseButton::Other(2),
    }
}
