use futures::{select, FutureExt};
use futures_timer::Delay;
use matchbox_socket::{PeerState, WebRtcSocket, PeerId};
use serde::{Deserialize, Serialize};
use std::{time::Duration, collections::HashMap};
use tokio::runtime::Runtime;
use std::str;

use crate::{rollback_manager::{PlayerAction, RollbackData, Player, PlayerAbility}, SPAWN_LOCATION, card_system::{BaseCard, CardManager}, settings_manager::Settings};

pub struct NetworkConnection {
    socket: WebRtcSocket,
    _runtime: Runtime,
    player_idx_map: HashMap<PeerId, usize>,
    packet_queue: Vec<NetworkPacket>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum NetworkPacket {
    Action(u64, PlayerAction),
    DeckUpdate(u64, Vec<BaseCard>),
    DeltatimeUpdate(u64, f32),
}

impl NetworkConnection {
    pub fn new(settings: &Settings) -> Self {
        let room_url = if settings.is_remote {
            settings.remote_url.clone()
        } else {
            settings.local_url.clone()
        };
        println!("connecting to matchbox server: {:?}", room_url);
        let (socket, loop_fut) = WebRtcSocket::new_reliable(room_url);

        let loop_fut = loop_fut.fuse();
        let rt = Runtime::new().unwrap();

        rt.spawn(async move {
            let timeout = Delay::new(Duration::from_millis(50));
            futures::pin_mut!(loop_fut, timeout);
            loop {
                select! {
                    _ = (&mut timeout).fuse() => {
                        timeout.reset(Duration::from_millis(50));
                    }

                    _ = &mut loop_fut => {
                        break;
                    }
                }
            }
        });

        NetworkConnection {
            socket,
            _runtime: rt,
            player_idx_map: HashMap::new(),
            packet_queue: Vec::new(),
        }
    }

    pub fn queue_packet(&mut self, packet: NetworkPacket) {
        self.packet_queue.push(packet);
    }

    pub fn network_update(&mut self, player_action: &PlayerAction, player_cards: &Vec<BaseCard>, card_system: &mut CardManager, rollback_manager: &mut RollbackData) {
        // Build message to send to peers
        let packet_data = NetworkPacket::Action(rollback_manager.current_time, player_action.clone());
        let packet = ron::to_string(&packet_data).unwrap().as_bytes().to_vec().into_boxed_slice();

        // Handle any new peers
        for (peer, state) in self.socket.update_peers() {
            match state {
                PeerState::Connected => {
                    println!("Peer joined: {:?}", peer);

                    self.player_idx_map.insert(peer, rollback_manager.rollback_state.players.len());

                    rollback_manager.player_join(Player {
                        pos: SPAWN_LOCATION,
                        ..Default::default()
                    });

                    {
                        let deck_packet_data = NetworkPacket::DeckUpdate(rollback_manager.current_time, player_cards.clone());
                        let deck_packet = ron::to_string(&deck_packet_data).unwrap().as_bytes().to_vec().into_boxed_slice();
                        self.socket.send(deck_packet.clone(), peer);
                    }
                    {
                        let dt_packet_data = NetworkPacket::DeltatimeUpdate(rollback_manager.current_time, rollback_manager.delta_time);
                        let dt_packet = ron::to_string(&dt_packet_data).unwrap().as_bytes().to_vec().into_boxed_slice();
                        self.socket.send(dt_packet.clone(), peer);
                    }
                }
                PeerState::Disconnected => {
                    println!("Peer left: {:?}", peer);
                }
            }
            self.socket.send(packet.clone(), peer);
        }

        if rollback_manager.rollback_state.players.len() <= 1 {
            return;
        }

        for (peer, _) in self.player_idx_map.iter() {
            self.socket.send(packet.clone(), *peer);
            for packet in self.packet_queue.drain(..) {
                let packet = ron::to_string(&packet).unwrap().as_bytes().to_vec().into_boxed_slice();
                self.socket.send(packet.clone(), *peer);
            }
        }

        // Accept any messages incoming
        for (peer, packet) in self.socket.receive() {
            let message = str::from_utf8(packet.as_ref()).unwrap();
            let foreign_player_action:NetworkPacket = ron::from_str(message).unwrap();

            let player_idx = self.player_idx_map.get(&peer).unwrap().clone();
            match foreign_player_action {
                NetworkPacket::Action(time, action) => {
                    rollback_manager.send_action(action, player_idx, time);
                }
                NetworkPacket::DeckUpdate(time, cards) => {
                    rollback_manager.send_deck_update(cards, player_idx, time);
                }
                NetworkPacket::DeltatimeUpdate(time, delta_time) => {
                    rollback_manager.send_dt_update(delta_time, player_idx, time);
                }
            }
        }
    }
}