use bytemuck::{Pod, Zeroable};
use futures::{select, FutureExt};
use futures_timer::Delay;
use matchbox_socket::{PeerState, WebRtcSocket, PeerId};
use std::{time::Duration, collections::HashMap};
use tokio::runtime::Runtime;

use crate::{rollback_manager::{PlayerAction, RollbackData, Player}, SPAWN_LOCATION};

pub struct NetworkConnection {
    socket: WebRtcSocket,
    _runtime: Runtime,
    player_idx_map: HashMap<PeerId, usize>,
}

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct NetworkPacket {
    pub time: u64,
    pub action: PlayerAction,
}

impl NetworkConnection {
    pub fn new() -> Self {
        let room_url = "ws://127.0.0.1:3536/extreme_bevy?next=2";
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

    pub fn network_update(&mut self, player_action: &PlayerAction, rollback_manager: &mut RollbackData) {
        // Build message to send to peers
        let packet_data = NetworkPacket {
            time: rollback_manager.current_time,
            action: player_action.clone(),
        };
        let packet = bytemuck::bytes_of(&packet_data).to_vec().into_boxed_slice();

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
            let foreign_player_action = bytemuck::from_bytes::<NetworkPacket>(&packet);
            let player_idx = self.player_idx_map.get(&peer).unwrap().clone();

            rollback_manager.send_action(foreign_player_action.action, player_idx, foreign_player_action.time);
        }
    }
}