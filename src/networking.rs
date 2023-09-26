use futures::{select, FutureExt};
use futures_timer::Delay;
use matchbox_socket::{PeerState, WebRtcSocket, PeerId};
use serde::{Deserialize, Serialize};
use std::{time::Duration, collections::HashMap};
use tokio::runtime::Runtime;
use std::str;

use crate::{rollback_manager::{PlayerAction, RollbackData, Player}, SPAWN_LOCATION, card_system::{BaseCard, CardManager}};

pub struct NetworkConnection {
    socket: WebRtcSocket,
    _runtime: Runtime,
    player_idx_map: HashMap<PeerId, usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum NetworkPacket {
    Action(u64, PlayerAction),
    DeckUpdate(BaseCard),
}

const IS_REMOTE_SERVER: bool = false;

impl NetworkConnection {
    pub fn new() -> Self {
        let room_url = if IS_REMOTE_SERVER {
            "ws://primd.net:3536/extreme_bevy?next=2"
        } else {
            "ws://127.0.0.1:3536/extreme_bevy?next=2"
        };
        println!("connecting to matchbox server: {:?}", room_url);
        let (socket, loop_fut) = WebRtcSocket::new_reliable(room_url);

        let loop_fut = loop_fut.fuse();
        let rt = Runtime::new().unwrap();

        rt.spawn(async move {
            let timeout = Delay::new(Duration::from_millis(100));
            futures::pin_mut!(loop_fut, timeout);
            loop {
                select! {
                    _ = (&mut timeout).fuse() => {
                        timeout.reset(Duration::from_millis(100));
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
        }
    }

    pub fn network_update(&mut self, player_action: &PlayerAction, player_cards: &BaseCard, card_system: &mut CardManager, rollback_manager: &mut RollbackData) {
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

                    let deck_packet_data = NetworkPacket::DeckUpdate(player_cards.clone());
                    let deck_packet = ron::to_string(&deck_packet_data).unwrap().as_bytes().to_vec().into_boxed_slice();
                    self.socket.send(deck_packet.clone(), peer);
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
                NetworkPacket::DeckUpdate(card) => {
                    rollback_manager.rollback_state.players[player_idx].cards_reference = card_system.register_base_card(card);
                }
            }
        }
    }
}