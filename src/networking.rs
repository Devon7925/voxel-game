use futures::{select, FutureExt};
use futures_timer::Delay;
use matchbox_socket::{PeerId, PeerState, WebRtcSocket};
use serde::{Deserialize, Serialize};
use std::str;
use std::time::Duration;
use tokio::runtime::Runtime;

use crate::{card_system::Cooldown, rollback_manager::PlayerAction, settings_manager::Settings, game_manager::GameSettings};

#[derive(Debug)]
pub struct NetworkConnection {
    socket: WebRtcSocket,
    _runtime: Runtime,
    packet_queue: Vec<NetworkPacket>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum NetworkPacket {
    Action(u64, PlayerAction),
    DeckUpdate(u64, Vec<Cooldown>),
    DeltatimeUpdate(u64, f32),
}

impl NetworkConnection {
    pub fn new(settings: &Settings, game_settings: &GameSettings) -> Self {
        let room_url = format!("{}{}", if game_settings.is_remote {
            settings.remote_url.clone()
        } else {
            settings.local_url.clone()
        }, game_settings.player_count);
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
            packet_queue: Vec::new(),
        }
    }

    pub fn queue_packet(&mut self, packet: NetworkPacket) {
        self.packet_queue.push(packet);
    }

    pub fn network_update(
        &mut self,
        player_count: usize,
    ) -> (Vec<(PeerId, PeerState)>, Vec<(PeerId, NetworkPacket)>) {
        let mut player_connection_changes = Vec::new();
        let mut recieved_packets = Vec::new();
        // Handle any new peers
        for (peer, state) in self.socket.update_peers() {
            player_connection_changes.push((peer, state));
        }

        if player_count > 1 {
            // Accept any messages incoming
            for (peer, packet) in self.socket.receive() {
                let message = str::from_utf8(packet.as_ref()).unwrap();
                let foreign_player_action: NetworkPacket = ron::from_str(message).unwrap();
                recieved_packets.push((peer, foreign_player_action));
            }
        }
        (player_connection_changes, recieved_packets)
    }

    pub fn send_packet(&mut self, peer: PeerId, packet: NetworkPacket) {
        let packet = ron::to_string(&packet)
            .unwrap()
            .as_bytes()
            .to_vec()
            .into_boxed_slice();
        self.socket.send(packet.clone(), peer);
    }

    pub fn send_packet_queue(&mut self, peers: Vec<&PeerId>) {
        for peer in peers {
            for packet in self.packet_queue.clone() {
                let packet = ron::to_string(&packet)
                    .unwrap()
                    .as_bytes()
                    .to_vec()
                    .into_boxed_slice();
                self.socket.send(packet.clone(), peer.clone());
            }
        }
        self.packet_queue.clear();
    }
}
